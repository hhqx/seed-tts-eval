import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]


model_id = "openai/whisper-large-v3"

# infer speed: # 2.64 sample / second
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
) 

# infer speed: # 2.20 sample / second
# model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, attn_implementation="flash_attention_2")


print(device)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    batch_size=16,  # batch size for inference - set based on your device
    torch_dtype=torch_dtype,
    device=device,
)


# result = pipe(sample)
# print(result["text"])

# import glob
# from tqdm import tqdm


# data_folder = "datasets/audio/LLM_ad7dbf22"

# dataset = load_dataset(data_folder, split="test", data_files={"test": "text_wav_LLM_ad7dbf22.json"})
# print(f"Total samples: {len(dataset)}")
# for item in tqdm(dataset):
#     sample = item["wav_path"]
    
#     sample_wav = os.path.join(data_folder, os.path.basename(item["wav_path"]))
#     result = pipe(sample_wav)
#     print(f"Sample: {item['wav_path']}")
#     print(f"Expected: {item['text']}")
#     print(result["text"])


import os
import time
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
# breakpoint()

# 配置路径
data_folder = "datasets/audio/LLM_ad7dbf22"

# 加载数据集
start_time = time.time()
dataset = load_dataset(
    data_folder, 
    split="test", 
    data_files={"test": "text_wav_LLM_ad7dbf22.json"}
)
load_time = time.time() - start_time

# 打印数据集信息
print(f"\n{'='*60}")
print(f"数据集加载完成 | 时间: {datetime.now().strftime('%H:%M:%S')}")
print(f"{'='*60}")
print(f"总样本数: {len(dataset)} | 加载耗时: {load_time:.2f}秒")
print(f"{'='*60}\n")

# 存储处理时间用于统计
processing_times = []

# dataset = dataset.to_pandas().to_dict(orient="records")

# ------------- do asr ----------------------

# 处理每个样本
for i, item in enumerate(tqdm(dataset, desc="处理进度")):
    # 记录开始时间
    item_start = time.time()
    
    # 构建音频文件路径
    wav_filename = os.path.basename(item["wav_path"])
    sample_wav = os.path.join(data_folder, wav_filename)
    
    # 处理音频
    result = pipe(sample_wav)
    
    item['wav_asr_text'] = result["text"]
    
    # [pipe(sample_wav) for _ in range(9)]
    
    # 计算处理时间
    item_time = time.time() - item_start
    processing_times.append(item_time)
    
    # 美化输出
    print(f"\n{'-'*50}")
    print(f"样本 #{i+1}/{len(dataset)} | 处理时间: {item_time:.2f}秒")
    print(f"时间戳: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'-'*50}")
    print(f"音频路径: {item['wav_path']}")
    print(f"{'-'*20}")
    print(f"预期文本: {item['text']}")
    print(f"{'-'*20}")
    print(f"识别结果: {result['text']}")
    print(f"{'-'*50}\n")

# 计算并显示总体统计信息
if processing_times:
    total_time = sum(processing_times)
    avg_time = total_time / len(processing_times)
    print(f"\n{'='*60}")
    print(f"所有样本处理完成 | 结束时间: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    print(f"总处理时间: {total_time:.2f}秒")
    print(f"平均处理时间: {avg_time:.2f}秒/样本")
    print(f"处理速度: {len(processing_times)/total_time:.2f}样本/秒")
    print(f"{'='*60}\n")


# ------------- cal wer ----------------------
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

# 2. 用map批量处理整个数据集（添加wer字段）
# 注意：map会返回新的Dataset，原数据集不会被修改
dataset_with_wer = dataset.map(
    process_example,
    num_proc=4  # 可选：多进程加速（根据CPU核心数调整）
)

# 3. 遍历处理后的数据集，打印结果（保持原输出格式）
for item in dataset_with_wer:
    print(f"样本: {item['wav_path']}")
    print(f"WER: {item['wer']:.2f}")
    print(f"预期文本: {item['text']}")
    print(f"识别结果: {item['wav_asr_text']}")
    print("-" * 50)
    print()