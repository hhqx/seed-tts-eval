import os
from datasets import load_dataset

repo_dir = "hhqx/seedtts_testset"

ds_en = load_dataset(repo_dir, 'en', trust_remote_code=True)
print(ds_en['test_wer'][0])
print(ds_en['test_wer'][0].keys())

ds_zh = load_dataset(repo_dir, 'zh', trust_remote_code=True)
print(ds_zh['test_sim'][0])


# Access specific splits
en_wer = ds_en['test_wer']
en_sim = ds_en['test_sim']

zh_wer = ds_zh['test_wer']
zh_sim = ds_zh['test_sim']
zh_hardcase = ds_zh['test_wer_hardcase']


for config, split in [
    ['en', 'test_wer'],
    ['en', 'test_sim'],
    ['zh', 'test_wer'],
    ['zh', 'test_sim'],
    ['zh', 'test_wer_hardcase'],
]:
    data = load_dataset(repo_dir, config, trust_remote_code=True, split=split)
    for item in data:
        for key, value in item.items():
            if key in ['audio_ground_truth', 'prompt_audio', ] and value:
                assert os.path.exists(value), f'path not exist: {value}'
    print("len of {} {}: {}".format(config, split, len(data)))