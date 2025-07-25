# WER

REPO_SRC=root/hqx/llm-tts
export PYTHONPATH=$REPO_SRC:$PYTHONPATH
echo 'add $REPO_SRC to PYTHONPATH'

## run model infer of seed-tts-eval 

OUTPUT_WAV_DIR=$1
if not OUTPUT_WAV_DIR:
  echo "you must"

## cal word error rate
bash cal_wer.sh /root/hqx/eval/seed-tts-eval/datasets/seedtts_testset/en/meta.lst \
  $OUTPUT_WAV_DIR \
  en

