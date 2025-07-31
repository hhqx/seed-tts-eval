import os
import json
import torch
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm
from typing import List, Callable, Dict, Optional
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from evaluate import load
from py3_tools.py_debug import breakpoint

import string
from jiwer import compute_measures, wer
from zhon.hanzi import punctuation

punctuation_all = punctuation + string.punctuation

def wer_seedtts(predictions, references, lang="en"):
    if isinstance(predictions, list):
        assert isinstance(references, list) and len(predictions) == len(references)
        return sum([wer_seedtts(pred, ref, lang) for pred, ref in zip(predictions, references)]) / len(predictions)
    
    truth, hypo = references, predictions
    breakpoint()
    
    if not predictions:
        return None  # 如果没有参考文本，返回 None

    for x in punctuation_all:
        if x == '\'':
            continue
        truth = truth.replace(x, '')
        hypo = hypo.replace(x, '')

    truth = truth.replace('  ', ' ')
    hypo = hypo.replace('  ', ' ')

    if lang == "zh":
        truth = " ".join([x for x in truth])
        hypo = " ".join([x for x in hypo])
    elif lang == "en":
        truth = truth.lower()
        hypo = hypo.lower()
    else:
        raise NotImplementedError

    measures = compute_measures(truth, hypo)
    wer = measures["wer"]
    breakpoint()
    return wer

# 加载WER评估指标
wer = load("wer")
def wer_huggingface(predictions, references):
    """使用Hugging Face的wer计算"""
    return wer.compute(
        predictions=predictions,
        references=references
    )


def process_example(example):
    """处理单个样本，计算WER"""
    example["wav_asr_text"] = example["wav_asr_text"].strip() if example["wav_asr_text"] else ""
    example["text"] = example["text"].strip() if example["text"] else ""
    
    example["wer"] = wer_huggingface(
        predictions=[example["wav_asr_text"]],
        references=[example["text"]]
    )
    example["wer"] = wer_seedtts(
        predictions=[example["wav_asr_text"]],
        references=[example["text"]]
    )
    return example

import pandas as pd 
def cal_wer(
    # json_file: str = 'outputs/seedtts-eval/spark-tts/LLM_ad7dbf22/seedtts/wer_asr_results.jsonl',
    json_file: str = 'outputs/seedtts-eval/spark-tts/exp0719_4090x6_constan_lr5e5_perbs256_ds2_6epoch_6417a43a/seedtts/wer_asr_results.jsonl',
    ):
    df = pd.read_json(json_file, lines=True)
    
    df['wer'] = df.apply(lambda x: wer_seedtts(x['wav_asr_text'], x['text']), axis=1)
    
    print(f"WER Mean: {df['wer'].mean()}")
    
# cal_wer()
# breakpoint()


def load_pipe(rank, model_id="openai/whisper-large-v3"):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration 

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    # processor = AutoProcessor.from_pretrained(model_id)
    # model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    # model.to(device)
    
    # # 创建pipeline
    # pipe = pipeline(
    #     "automatic-speech-recognition",
    #     model=model,
    #     tokenizer=processor.tokenizer,
    #     feature_extractor=processor.feature_extractor,
    #     device=device,
    #     chunk_length_s=30
    # )

    # model_id = "openai/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    
    
    return model, processor

import soundfile as sf
import scipy
def do_asr(pipe, example):
    model, processor = pipe
    device = next(model.parameters()).device

    wav, sr = sf.read(example["wav_path"])
    if sr != 16000:
        wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
    input_features = processor(wav, sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.strip()

def whisper_asr_worker(pipe, example):
    """ASR处理函数，每个进程独立初始化模型"""
    # 初始化处理器和模型
    example['text'] = example.pop('text')
    
    # 执行ASR推理
    try:

        asr_txt = do_asr(pipe, example)
        example["wav_asr_text"] = asr_txt
    except Exception as e:
        example["wav_asr_text"] = None
        example["error"] = str(e)
        print(f"Error processing {example['wav_path']}: {str(e)}")
        
    # 计算WER（如果ASR成功）
    if example["wav_asr_text"] is not None:
        example = process_example(example)
    else:
        example["wer"] = None  # 标记错误样本的WER
    
    return example

def main_worker(rank, args):
    """每个进程的主函数"""
    data, gpu_ids, extra_args, results, errors, lock, counter = args
    ngpus = len(gpu_ids)
    device = torch.device(f"cuda:{rank}")  # rank对应GPU ID
    
    pipe = load_pipe(rank)
    
    # 数据分片：每个GPU处理 data[rank::ngpus]
    local_data = data[rank::ngpus]
    print(f"Rank {rank} 开始处理 {len(local_data)} 个样本")
    
    # 处理本地数据
    for idx, item in enumerate(tqdm(local_data, desc=f"Rank {rank} 进度")):
        # 原始数据索引（用于结果排序）
        original_idx = item["original_idx"]
        try:
            # 调用ASR处理函数
            result = whisper_asr_worker(pipe, item)
            # 保存结果（带原始索引）
            with lock:
                results[original_idx] = result
                counter.value += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            # 记录错误
            with lock:
                print("rank", rank, "error:", str(e))
                errors.append((original_idx, item, str(e)))
    
    print(f"Rank {rank} 处理完成")


def run_multi_gpu_tasks(
    data: List,
    worker_fn: Callable,
    gpu_ids: List[int],
    extra_args: Dict = {},
    output_file: Optional[str] = None,
    error_file: Optional[str] = None,
    desc: str = "任务进度"
):
    # 为每个样本添加原始索引，用于结果排序
    data_with_idx = [{"original_idx": i,** item} for i, item in enumerate(data)]
    total = len(data)
    
    # 初始化共享内存：结果列表（预分配空间）、错误列表、计数器
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    results = manager.list([None] * total)  # 固定长度，按原始索引存储
    errors = manager.list()
    counter = manager.Value('i', 0)
    lock = manager.Lock()
    
    # 准备传给每个worker的参数
    args = (
        data_with_idx,
        gpu_ids,
        extra_args,
        results,
        errors,
        lock,
        counter
    )
    # main_worker(0, args)  # 主进程先执行一次，加载模型等
    
    if 1 or not os.path.exists(output_file):
        # 启动多进程（每个GPU一个进程）
        try:
            # 使用spawn启动进程，nprocs=len(gpu_ids)，每个进程对应一个rank（0到ngpus-1）
            torch.multiprocessing.spawn(
                main_worker,
                args=(args,),
                nprocs=len(gpu_ids),
                join=True
            )
        except KeyboardInterrupt:
            print("🛑 手动中断")
            raise
        except Exception as e:
            print(f"❌ 进程错误: {str(e)}")
            raise
    
        # 转换共享列表为普通列表
        results = list(results)
        errors = list(errors)
    else:
        results = pd.read_json(output_file, lines=True).to_dict(orient="records")
        # error_file = pd.read_json(error_file, lines=True).to_dict(orient="records")
        
        for _ in results:
            if np.isnan(_["wer"]):
                _["wer"] = None

    print("✅ 汇总结果...")
    
    breakpoint()
    
    # 保存输出文件
    if output_file:
        results.sort(key=lambda x: -x["wer"] if x["wer"] is not None else float('inf'))  # 按WER排序

        pd.DataFrame(results).to_csv(output_file.replace(".jsonl", ".csv"), index=False, header=True)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            for res in results:
                if res is not None:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
    
    # 保存错误文件
    # if error_file and errors:
    #     os.makedirs(os.path.dirname(error_file), exist_ok=True)
    #     with open(error_file, "w") as f:
    #         for e in errors:
    #             f.write(json.dumps({
    #                 "original_idx": e[0],
    #                 "item": e[1],
    #                 "error": e[2]
    #             }, ensure_ascii=False) + "\n")
    
    # 计算有效样本的WER均值
    
    valid_wer = [res["wer"] for res in results if res is not None and res["wer"] is not None]
    print(f"✅ 成功: {len(valid_wer)}/{total}")
    print(f"❌ 失败: {len(errors)}")
    if valid_wer:
        print(f"wer mean: {np.mean(valid_wer)}")
    
    return results, errors

def main(data_folder):
    import glob
    import json
    import numpy as np
    
    # # # sparktts, has_ref, result: 0.0733, seedtts-eval: 0.0253
    # # '/home/hqx/llm-tts/outputs/seedtts-eval/spark-tts/LLM_ad7dbf22',
    
    # # pretrain, one-shot, has_ref, result: 0.0843, seedtts-eval: 0.0314
    # '/home/hqx/llm-tts/outputs/seedtts-eval/spark-tts/exp0719_4090x6_constan_lr5e5_perbs256_ds2_6epoch_6417a43a',
    
    # # Pretrain，zero-shot, no_ref, result: xxx, seedtts-eval: 0.0298
    # '/home/hqx/llm-tts/outputs/seedtts-eval/spark-tts/exp0719_4090x6_constan_lr5e5_perbs256_ds2_6epoch_1853c44b',
    
    
    # # Original repo, one-shot, has_ref, result: xxx, seedtts-eval: 0.0244
    # '/home/hqx/Spark-TTS/example/seedtts-results',
    
    # # Original repo, zero-shot, no_ref, result: xxx, seedtts-eval: xxx
    # '/home/hqx/Spark-TTS/example/seedtts-results-male_moderate_moderate',

    
    output_folder = os.path.join(data_folder, "seedtts")
    os.makedirs(output_folder, exist_ok=True)
    
    json_files = glob.glob(os.path.join(data_folder, "text_wav_*.json"))
    if len(json_files) != 1:
        raise ValueError(f"Expected exactly one JSON file in {data_folder}, found {len(json_files)}")
    input_json = json_files[0]
    output_file = os.path.join(output_folder, "wer_asr_results.jsonl")
    error_file = os.path.join(output_folder, "wer_asr_failed_samples.jsonl")
    
    print(f"加载数据集: {input_json}")
    with open(input_json) as f:
        data = json.load(f)
    
    # 运行多GPU任务

    ngpu = torch.cuda.device_count()
    print(f"检测到 {ngpu} 个可用GPU")
    results, errors = run_multi_gpu_tasks(
        data=data,
        worker_fn=whisper_asr_worker,
        # gpu_ids=[0],  # 使用的GPU ID列表
        gpu_ids=list(range(ngpu)),  # 自动检测所有GPU
        extra_args={"model_id": "openai/whisper-large-v3"},
        output_file=output_file,
        error_file=error_file,
        desc="Whisper 多GPU ASR推理"
    )
    
    # 计算并打印平均WER（过滤错误样本）
    valid_wer = [res["wer"] for res in results if res is not None and res["wer"] is not None]
    if valid_wer:
        print(f"最终WER均值: {np.mean(valid_wer):.4f}")
    else:
        print("没有有效样本计算WER")
    
    with open(output_file.replace('.jsonl', '.log'), "w") as f:
        f.write(f"最终WER均值: {np.mean(valid_wer):.4f}\n")
        f.write(f"成功样本数: {len(valid_wer)}\n")
        f.write(f"失败样本数: {sum(res['wer'] is None for res in results)}\n")


# 主程序入口
if __name__ == "__main__":
    for data_folder in [
        # # sparktts, one-shot, has_ref,  seedtts-eval: 0.0246
        '/home/hqx/llm-tts/outputs/seedtts-eval/spark-tts/LLM_ad7dbf22',
        
        # # pretrain, one-shot, has_ref, seedtts-eval: 0.0314
        # '/home/hqx/llm-tts/outputs/seedtts-eval/spark-tts/exp0719_4090x6_constan_lr5e5_perbs256_ds2_6epoch_6417a43a',
        
        # # Pretrain，zero-shot, no_ref, seedtts-eval: 0.0298
        # '/home/hqx/llm-tts/outputs/seedtts-eval/spark-tts/exp0719_4090x6_constan_lr5e5_perbs256_ds2_6epoch_1853c44b',
        
        
        # # Original repo, one-shot, has_ref,  seedtts-eval: 0.0244
        # '/home/hqx/Spark-TTS/example/seedtts-results',
        
        # # Original repo, zero-shot, no_ref,  seedtts-eval: 0.0249
        # '/home/hqx/Spark-TTS/example/seedtts-results-male_moderate_moderate',
    ]:
        print(f"处理数据文件夹: {data_folder}")
        main(data_folder)
 