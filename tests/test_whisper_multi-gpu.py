import os
import time
import torch
from datasets import load_dataset
from tqdm import tqdm
from multiprocessing import current_process

import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

# 假设 pipe 是你的 ASR 处理函数（如 Whisper/Paraformer 推理接口）
# 注意：pipe 需要支持在指定 GPU 上运行

def load_pipe(device):
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    # infer speed: # 2.64 sample / second
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    ) 

    # infer speed: # 2.20 sample / second
    # model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, attn_implementation="flash_attention_2")


    # print(device)
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
    return pipe

def init_process(gpu_ids):
    """初始化进程：为每个子进程分配专属 GPU"""
    # 获取当前进程 ID（0, 1, ..., num_proc-1）
    proc_id = int(current_process().name.split('-')[-1])
    # 分配对应的 GPU（确保 gpu_ids 数量 >= num_proc）
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[proc_id])
    # 验证 GPU 分配
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"进程 {proc_id} 绑定 GPU: {gpu_ids[proc_id]}，设备: {device}")
    return device

import os
import torch
from multiprocessing import current_process
import psutil  # 用于获取子进程的父进程关系（需安装：pip install psutil）

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

def init_process(gpu_ids):
    """初始化进程：为每个子进程分配专属 GPU（修复主进程解析问题）"""
    # 获取当前进程的 PID
    current_pid = os.getpid()
    # 获取主进程 PID（启动所有子进程的父进程）
    main_pid = os.getppid()

    # 主进程不绑定 GPU（仅子进程需要）
    if current_pid == main_pid:
        print("主进程不绑定 GPU")
        return torch.device("cpu")  # 主进程返回 CPU 即可

    # 子进程：获取所有子进程的列表，确定当前进程的索引
    # 1. 找到主进程的所有子进程（即我们通过 num_proc 启动的进程）
    main_process = psutil.Process(main_pid)
    child_processes = main_process.children(recursive=False)  # 仅一级子进程
    child_pids = [p.pid for p in child_processes]

    # 2. 确定当前子进程在列表中的索引（即 proc_id）
    proc_id = child_pids.index(current_pid)

    # 3. 分配对应的 GPU（确保不超过 gpu_ids 范围）
    if proc_id < len(gpu_ids):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[proc_id])
        device = torch.device("cuda")
        print(f"子进程 {proc_id}（PID: {current_pid}）绑定 GPU: {gpu_ids[proc_id]}")
    else:
        # 若进程数超过 GPU 数，默认使用 CPU（可根据需求调整）
        device = torch.device("cpu")
        print(f"子进程 {proc_id}（PID: {current_pid}）未分配 GPU，使用 CPU")

    return device

def process_item(example, data_folder, gpu_ids):
    """单样本处理函数：在指定 GPU 上运行 ASR"""
    # 初始化当前进程的 GPU（每个进程仅初始化一次）
    if not hasattr(process_item, "device"):
        process_item.device = init_process(gpu_ids)
        # 每个进程独立加载模型（避免多进程冲突）
        # 注意：这里需要根据你的 pipe 实现加载模型到 process_item.device
        global pipe
        pipe = load_pipe(device=process_item.device)  # 加载模型到当前进程的 GPU

    # 构建音频文件路径
    wav_filename = os.path.basename(example["wav_path"])
    sample_wav = os.path.join(data_folder, wav_filename)
    
    # 处理音频（使用当前进程绑定的 GPU）
    result = pipe(sample_wav)
    example['wav_asr_text'] = result["text"]
    
    return example


def main(data_folder, gpu_ids=[0, 1, 2, 3]):
    # 加载数据集
    dataset = load_dataset(
        data_folder, 
        split="test", 
        data_files={"test": "text_wav_LLM_ad7dbf22.json"}
    )
    print(f"数据集大小: {len(dataset)}")

    # 记录开始时间
    start_time = time.time()

    # 使用 map 多进程并行处理（每个进程绑定一个 GPU）
    processed_dataset = dataset.map(
        process_item,
        fn_kwargs={  # 传递额外参数给 process_item
            "data_folder": data_folder,
            "gpu_ids": gpu_ids
        },
        num_proc=len(gpu_ids),  # 进程数 = GPU 数
        desc="多 GPU 并行 ASR 处理"
    )

    # 计算总耗时
    total_time = time.time() - start_time
    print(f"总处理时间: {total_time:.2f} 秒")
    print(f"平均处理速度: {len(dataset)/total_time:.2f} 样本/秒")

    return processed_dataset


if __name__ == "__main__":
    # 示例：使用 GPU 0,1,2,3 并行处理
    processed_ds = main(
        data_folder="datasets/audio/LLM_ad7dbf22",
        gpu_ids=[0, 1, 2, 3]
    )
    # 保存处理结果
    processed_ds.save_to_disk("outputs/processed_asr_results")