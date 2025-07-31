import os
import sys
import argparse
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="WER计算处理脚本")
    parser.add_argument("meta_lst", help="元数据列表文件路径")
    parser.add_argument("output_dir", help="输出目录路径")
    parser.add_argument("lang", help="语言参数")
    return parser.parse_args()

def get_working_dir():
    """获取工作目录（脚本所在目录的上一级）"""
    script_path = Path(os.path.abspath(__file__))
    return script_path.parent.parent.resolve()

def prepare_directories(thread_dir, out_dir):
    """创建必要的目录"""
    Path(thread_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

def split_input_file(input_file, thread_dir, num_per_thread):
    """
    将输入文件分割成多个部分，类似于bash的split命令
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 分割文件并写入
    for i in range(0, len(lines), num_per_thread):
        chunk = lines[i:i+num_per_thread]
        chunk_file = Path(thread_dir) / f"thread-{i//num_per_thread:02d}.lst"
        with open(chunk_file, 'w', encoding='utf-8') as f:
            f.writelines(chunk)
    
    return len(lines) // num_per_thread + (1 if len(lines) % num_per_thread else 0)


def process_chunk(args):
    """处理单个文件块的函数，用于多进程调用（实时输出版本）"""
    rank, thread_file, sub_score_file, lang = args
    # 设置CUDA可见设备
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(rank)
    
    print(f"进程 {rank} 开始处理: {thread_file}")
    
    # 调用外部脚本并实时输出
    # 通过stdout=sys.stdout和stderr=sys.stderr实现实时输出
    result = subprocess.run(
        ["python3", "run_wer.py", str(thread_file), str(sub_score_file), lang],
        env=env,
        stdout=sys.stdout,  # 实时输出stdout到当前控制台
        stderr=sys.stderr,  # 实时输出stderr到当前控制台
        text=True           # 确保输出为文本模式而非字节流
    )
    
    # 输出进程结束信息
    if result.returncode == 0:
        print(f"进程 {rank} 处理完成，结果保存至: {sub_score_file}")
    else:
        print(f"进程 {rank} 处理失败，返回码: {result.returncode}", file=sys.stderr)
    
    return result.returncode

import torch

def main():
    # 解析参数
    args = parse_args()
    meta_lst = args.meta_lst
    output_dir = args.output_dir
    lang = args.lang
    
    # 定义文件路径
    wav_wav_text = os.path.join(output_dir, "wav_res_ref_text")
    score_file = os.path.join(output_dir, "wav_res_ref_text.wer")
    
    # 获取工作目录
    workdir = get_working_dir()
    print(f"工作目录: {workdir}")
    
    # 第一步：生成wav_res_ref_text文件
    print("正在生成wav_res_ref_text文件...")
    result = subprocess.run(
        ["python3", "get_wav_res_ref_text.py", meta_lst, output_dir, wav_wav_text],
        check=True
    )
    if result.returncode != 0:
        print("生成wav_res_ref_text文件失败", file=sys.stderr)
        sys.exit(-1)
    
    # 配置GPU参数
    # cuda_visible_devices = [4, 5, 6, 7]  # GPU设备列表
    # cuda_visible_devices = [4]  # GPU设备列表
    num_gpu = torch.cuda.device_count()
    print(f"使用GPU数量: {num_gpu}")
    
    # 创建临时目录
    timestamp = int(time.time())
    thread_dir = f"/tmp/thread_metas_{timestamp}/"
    # thread_dir = 'tmp/debug'
    out_dir = os.path.join(thread_dir, "results/")
    prepare_directories(thread_dir, out_dir)
    print(f"临时工作目录: {thread_dir}")
    
    # 计算文件分割参数
    with open(wav_wav_text, 'r', encoding='utf-8') as f:
        num_lines = sum(1 for _ in f)
    print(f"总处理行数: {num_lines}")
    
    num_per_thread = num_lines // num_gpu + 1
    print(f"每个进程处理行数: {num_per_thread}")
    
    # 分割输入文件
    num_chunks = split_input_file(wav_wav_text, thread_dir, num_per_thread)
    print(f"文件分割为 {num_chunks} 个部分")
    
    # 准备并行处理的参数
    process_args = []
    for rank in range(num_gpu):
        thread_file = Path(thread_dir) / f"thread-{rank:02d}.lst"
        if not thread_file.exists():
            continue
            
        sub_score_file = os.path.join(out_dir, f"thread-0{rank}.wer.out")
        process_args.append((rank, thread_file, sub_score_file, lang))
    
    
    if len(process_args) == 1:
        print("仅有一个处理任务，直接执行...")
        results = [process_chunk(process_args[0])]  # 调试时只处理第一个文件
    else:
        # 使用多进程池处理文件
        print("开始并行处理...")
        with Pool(processes=num_gpu) as pool:
            results = pool.map(process_chunk, process_args)
    
    # 检查是否有错误
    if any(code != 0 for code in results):
        print("部分进程执行失败", file=sys.stderr)
        sys.exit(-1)
    
    # 合并结果
    print("合并处理结果...")
    merge_file = os.path.join(out_dir, "merge.out")
    with open(merge_file, 'w', encoding='utf-8') as outf:
        for rank in range(num_gpu):
            sub_file = os.path.join(out_dir, f"thread-0{rank}.wer.out")
            if os.path.exists(sub_file):
                with open(sub_file, 'r', encoding='utf-8') as inf:
                    outf.write(inf.read())
    
    # 计算平均WER
    print("计算最终WER分数...")
    subprocess.run(
        ["python3", "average_wer.py", merge_file, score_file],
        check=True
    )
    
    print("所有处理完成")

if __name__ == "__main__":
    main()
    