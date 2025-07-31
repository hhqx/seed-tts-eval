# multi_gpu_runner.py
import os
import time
import signal
import json
import torch
from multiprocessing import Process, Queue, Value, Lock
from tqdm import tqdm
from typing import List, Callable, Dict, Optional


def _worker_fn_wrapper(worker_fn, gpu_id, task_queue, result_queue, error_queue, counter, lock, extra_args):
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # 忽略 Ctrl+C

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"[进程 {os.getpid()}] 使用 GPU {device}")

    while not task_queue.empty():
        try:
            idx, item = task_queue.get(timeout=1)
        except:
            break

        try:
            result = worker_fn(item, gpu_id=gpu_id, device=device, **extra_args)
            result_queue.put((idx, result))
        except Exception as e:
            error_queue.put((idx, item, str(e)))

        with lock:
            counter.value += 1


def run_multi_gpu_tasks(
    data: List,
    worker_fn: Callable,
    gpu_ids: List[int],
    extra_args: Dict = {},
    output_file: Optional[str] = None,
    error_file: Optional[str] = None,
    desc: str = "任务进度"
):
    from multiprocessing import set_start_method
    set_start_method("spawn", force=True)

    task_queue = Queue()
    result_queue = Queue()
    error_queue = Queue()
    total = len(data)

    for idx, item in enumerate(data):
        task_queue.put((idx, item))

    counter = Value('i', 0)
    lock = Lock()

    processes = []
    for gpu_id in gpu_ids:
        p = Process(
            target=_worker_fn_wrapper,
            args=(worker_fn, gpu_id, task_queue, result_queue, error_queue, counter, lock, extra_args)
        )
        p.start()
        processes.append(p)

    try:
        with tqdm(total=total, desc=desc) as pbar:
            last = 0
            while any(p.is_alive() for p in processes):
                with lock:
                    now = counter.value
                pbar.update(now - last)
                last = now
                time.sleep(0.5)

            with lock:
                pbar.update(counter.value - last)
    except KeyboardInterrupt:
        print("🛑 手动中断，正在终止子进程...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
        raise

    for p in processes:
        p.join()

    print("✅ 汇总结果...")

    results = [None] * total
    while not result_queue.empty():
        idx, res = result_queue.get()
        results[idx] = res

    errors = []
    while not error_queue.empty():
        idx, ex, err = error_queue.get()
        errors.append({"index": idx, "input": ex, "error": err})

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            for item in results:
                if item is not None:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if error_file and errors:
        os.makedirs(os.path.dirname(error_file), exist_ok=True)
        with open(error_file, "w") as f:
            for e in errors:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"✅ 成功: {len([r for r in results if r is not None])}")
    print(f"❌ 失败: {len(errors)}")

    return results, errors

from evaluate import load
from datasets import Dataset  # 确保数据集是Dataset对象

# 加载WER评估指标
wer = load("wer")

# 1. 定义处理单个样本的函数（用于map）
def process_example(example):
    """
    处理单个样本：清洗文本、计算WER、添加到样本字段中
    """
    # 清洗文本（去除首尾空格）
    example["wav_asr_text"] = example["wav_asr_text"].strip()
    example["text"] = example["text"].strip()
    
    # 计算WER（保持与原逻辑一致，用列表包裹单样本）
    example["wer"] = wer.compute(
        predictions=[example["wav_asr_text"]],
        references=[example["text"]]
    )
    
    return example  # 返回处理后的样本


# whisper_runner.py
import json
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# from multi_gpu_runner import run_multi_gpu_tasks


# 自定义任务函数（必须支持 gpu_id, device）
def whisper_asr_worker(example, gpu_id, device, model_id):
    pipe = whisper_asr_worker.pipes.get(gpu_id)
    if pipe is None:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
        model.to(device)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=gpu_id,
            chunk_length_s=30
        )
        whisper_asr_worker.pipes[gpu_id] = pipe

    # asr
    # if not os.path.exists(example["wav_path"]) or not os.path.isfile(example["wav_path"]):
    #     Warning(f"音频文件不存在: {example['wav_path']}")
    #     example["wav_asr_text"] = None  # 如果音频文件不存在，ASR 文本设置为 None"
    #     example["wer"] = 1.0  # 如果音频文件不存在，WER
    # else:
    result = pipe(example["wav_path"])
    example["wav_asr_text"] = result["text"]
    
    # wer
    # 清洗文本（去除首尾空格）
    example["wav_asr_text"] = example["wav_asr_text"].strip()
    example["text"] = example["text"].strip()
    
    # 计算WER（保持与原逻辑一致，用列表包裹单样本）
    example["wer"] = wer.compute(
        predictions=[example["wav_asr_text"]],
        references=[example["text"]]
    )
    
    return example


# 初始化 pipe 缓存容器
whisper_asr_worker.pipes = {}


import glob
import pandas as pd
import numpy as np
if __name__ == "__main__":
    
    data_folder = '/home/hqx/llm-tts/outputs/seedtts-eval/spark-tts/exp0719_4090x6_constan_lr5e5_perbs256_ds2_6epoch_1853c44b'
    # data_folder = 'datasets/audio/LLM_ad7dbf22'
    output_folder = os.path.join(data_folder, "seedtts")
    os.makedirs(output_folder, exist_ok=True)
    
    input_json = glob.glob(os.path.join(data_folder, "text_wav_*.json"))[0]
    output_file = os.path.join(output_folder, "wer_asr_results.jsonl")
    error_file = os.path.join(output_folder, "wer_asr_failed_samples.jsonl")

    print(f"加载数据集: {input_json}")
    with open(input_json) as f:
        data = json.load(f)

    results, errors = run_multi_gpu_tasks(
        data=data,
        worker_fn=whisper_asr_worker,
        # gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        gpu_ids=[ 4, 5, 6, 7],
        extra_args={"model_id": "openai/whisper-large-v3"},
        output_file=output_file,
        error_file=error_file,
        desc="Whisper 多 GPU ASR 推理"
    )
    
    
    results = filter(lambda x: x['wav_asr_text'] is not None, results)
    
    # df = pd.DataFrame(results)
    # print(df['wer'].mean())
    
    print('wer mean:', np.array([x['wer'] for x in results]).mean())
