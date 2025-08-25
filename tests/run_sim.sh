
# ---------------------------------------------------------------------------------------------------------------------
# # Spark-TTS, llm-tts repo, 空白样本个数：6
# OUTPUT_WAV_DIR="/home/hqx/llm-tts/outputs/seedtts-eval/spark-tts/sim/LLM_3036959a/wav"
# bash cal_sim.sh /mnt/datasets_processed/seedtts_testset/data/en_non_para_reconstruct_meta.lst  \
#   $OUTPUT_WAV_DIR \
#   /mnt/pretrained_models/wavlm_large_finetune.pth

# ASV: 0.57
# ASV-var: 0.01

# ---------------------------------------------------------------------------------------------------------------------
# Spark-TTS, original repo, 空白样本个数：13
OUTPUT_WAV_DIR="/home/hqx/Spark-TTS/outputs/seedtts/sim/wav"
bash cal_sim.sh /mnt/datasets_processed/seedtts_testset/data/en_non_para_reconstruct_meta.lst  \
  $OUTPUT_WAV_DIR \
  /mnt/pretrained_models/wavlm_large_finetune.pth

# ASV: 0.569
# ASV-var: 0.011


# ---------------------------------------------------------------------------------------------------------------------
# # Pretrain Ours， one-shot, 空白样本个数：69
# OUTPUT_WAV_DIR="/home/hqx/llm-tts/outputs/seedtts-eval/tts-pretrain/sim/exp0719_4090x6_constan_lr5e5_perbs256_ds2_6epoch_662bbde7/wav"
# bash cal_sim.sh /mnt/datasets_processed/seedtts_testset/data/en_non_para_reconstruct_meta.lst  \
#   $OUTPUT_WAV_DIR \
#   /mnt/pretrained_models/wavlm_large_finetune.pth

# ASV: 0.527
# ASV-var: 0.011

# ---------------------------------------------------------------------------------------------------------------------
# # Pretrain Ours， zero-shot, 空白样本个数：0
# OUTPUT_WAV_DIR="/home/hqx/llm-tts/outputs/seedtts-eval/tts-pretrain/sim/exp0719_4090x6_constan_lr5e5_perbs256_ds2_6epoch_7d9fc897/wav"
# bash cal_sim.sh /mnt/datasets_processed/seedtts_testset/data/en_non_para_reconstruct_meta.lst  \
#   $OUTPUT_WAV_DIR \
#   /mnt/pretrained_models/wavlm_large_finetune.pth

# ASV: 0.037
# ASV-var: 0.01


# ---------------------------------------------------------------------------------------------------------------------


# 加测了不同 dataset split 的 sim 和 wer 指标结果，结果发现不同的dataset split 对结果几乎没有影响