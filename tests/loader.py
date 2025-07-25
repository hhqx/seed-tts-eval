import json
from typing import List, Dict, Any
from pathlib import Path

class TTSDataLoader:
    def __init__(self):
        self.data: List[Dict[str, str]] = []
    
    def load_from_file(self, filepath: str) -> None:
        """Load data from pipe-delimited file"""
        self.data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('|')
                    if len(parts) >= 4:
                        item = {
                            'synth_wavname': parts[0],
                            'prompt_text': parts[1],
                            'prompt_audio': parts[2],
                            'synthesis_text': parts[3],
                            'ground_truth_audio': parts[4] if len(parts) > 4 else None
                        }
                        self.data.append(item)
    
    def save_as_lst(self, output_path: str) -> None:
        """Save data as lst format (pipe-delimited)"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in self.data:
                line_parts = [
                    item['synth_wavname'],
                    item['prompt_text'],
                    item['prompt_audio'],
                    item['synthesis_text']
                ]
                if item['ground_truth_audio']:
                    line_parts.append(item['ground_truth_audio'])
                f.write('|'.join(line_parts) + '\n')
    
    def save_as_jsonl(self, output_path: str) -> None:
        """Save data as JSONL format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in self.data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def save_as_json(self, output_path: str) -> None:
        """Save data as JSON format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def get_data(self) -> List[Dict[str, str]]:
        """Return loaded data"""
        return self.data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.data[idx]

# Example usage
if __name__ == "__main__":
    loader = TTSDataLoader()
    
    # Load data from file
    loader.load_from_file('/root/hqx/eval/seed-tts-eval/datasets/seedtts_testset/en/meta.lst')
    loader.load_from_file('/root/hqx/eval/seed-tts-eval/datasets/seedtts_testset/en/non_para_reconstruct_meta.lst')
    
    # Save in different formats
    # loader.save_as_lst('output.lst')
    # loader.save_as_jsonl('output.jsonl')
    # loader.save_as_json('output.json')
    
    print(f"Loaded {len(loader)} items")
