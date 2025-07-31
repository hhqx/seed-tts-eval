# WER

REPO_SRC=root/hqx/llm-tts
export PYTHONPATH=$REPO_SRC:$PYTHONPATH
echo 'add $REPO_SRC to PYTHONPATH'

## run model infer of seed-tts-eval 

OUTPUT_WAV_DIR="outputs/wer"

## cal word error rate
# bash cal_wer.sh /nfs/datasets_processed/seedtts_testset/data/en_meta.lst  \
#   $OUTPUT_WAV_DIR \
#   en

# # spark-tts, one-shot, llmtts repo, 最终分数：WER: 2.461%
# OUTPUT_WAV_DIR="/home/hqx/llm-tts/outputs/seedtts-eval/spark-tts/LLM_ad7dbf22"
# python cal_wer.py /nfs/datasets_processed/seedtts_testset/data/en_meta.lst  \
#   $OUTPUT_WAV_DIR \
#   en

# # llm-tts, one-shot, pretrain on emilia_en WER: 3.136%
# OUTPUT_WAV_DIR="/home/hqx/llm-tts/outputs/seedtts-eval/spark-tts/exp0719_4090x6_constan_lr5e5_perbs256_ds2_6epoch_6417a43a/wav"
# python cal_wer.py /nfs/datasets_processed/seedtts_testset/data/en_meta.lst  \
#   $OUTPUT_WAV_DIR \
#   en


# # llm-tts pretrain, infer zero-shot, 最终分数：WER: 2.977%
# OUTPUT_WAV_DIR="outputs/seedtts-eval/spark-tts/exp0719_4090x6_constan_lr5e5_perbs256_ds2_6epoch_1853c44b/wav"
# python cal_wer.py /nfs/datasets_processed/seedtts_testset/data/en_meta.lst  \
#   $OUTPUT_WAV_DIR \
#   en

# # spark-tts, one-shot, original repo, 最终分数：WER: 2.449%;  WER: 2.565%； WER: 2.598%；
# Eval WER on Sim split dataset: WER: 2.956%
# OUTPUT_WAV_DIR="/home/hqx/Spark-TTS/example/seedtts-results/wav"
# python cal_wer.py /nfs/datasets_processed/seedtts_testset/data/en_meta.lst  \
#   $OUTPUT_WAV_DIR \
#   en

# #  spark-tts, zero-shot, original repo, 最终分数：WER: 2.49%
# OUTPUT_WAV_DIR="/home/hqx/Spark-TTS/example/seedtts-results-male_moderate_moderate/wav"
# python cal_wer.py /nfs/datasets_processed/seedtts_testset/data/en_meta.lst  \
#   $OUTPUT_WAV_DIR \
#   en



# # test bash for debug
# OUTPUT_WAV_DIR="/home/hqx/llm-tts/outputs/seedtts-eval/tts-pretrain/wer/exp0719_4090x6_constan_lr5e5_perbs256_ds2_6epoch_6417a43a/wav"
# bash cal_wer.sh /nfs/datasets_processed/seedtts_testset/data/en_meta.lst  \
#   $OUTPUT_WAV_DIR \
#   en


