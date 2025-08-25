# WER

# REPO_SRC=root/hqx/llm-tts
# export PYTHONPATH=$REPO_SRC:$PYTHONPATH
# echo 'add $REPO_SRC to PYTHONPATH'

## run model infer of seed-tts-eval 

## cal word error rate

# #  spark-tts, zero-shot, original repo, 最终分数：WER: 2.49%
OUTPUT_WAV_DIR="/home/hqx/llm-tts/outputs/higgs/seedtts_eval/rvq-ckpt-6epoch-ckpt-3.1epoch/wav/wav"
python cal_wer.py /mnt/datasets_processed/seedtts_testset/data/en_meta.lst  \
  $OUTPUT_WAV_DIR \
  en

OUTPUT_WAV_DIR="/home/hqx/llm-tts/outputs/higgs/seedtts_eval/rvq-ckpt-6epoch/wav/wav"
python cal_wer.py /mnt/datasets_processed/seedtts_testset/data/en_meta.lst  \
  $OUTPUT_WAV_DIR \
  en

OUTPUT_WAV_DIR="/home/hqx/llm-tts/outputs/higgs/seedtts_eval/rvq-ckpt-6-12-epoch-ckpt-latest/wav/wav"
python cal_wer.py /mnt/datasets_processed/seedtts_testset/data/en_meta.lst  \
  $OUTPUT_WAV_DIR \
  en


# # test bash for debug
# OUTPUT_WAV_DIR="/home/hqx/llm-tts/outputs/seedtts-eval/tts-pretrain/wer/exp0719_4090x6_constan_lr5e5_perbs256_ds2_6epoch_6417a43a/wav"
# bash cal_wer.sh /mnt/datasets_processed/seedtts_testset/data/en_meta.lst  \
#   $OUTPUT_WAV_DIR \
#   en


