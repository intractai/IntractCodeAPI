import json
from datasets import load_dataset

def download_xcode_eval_dataset():
    #dataset = load_dataset("deepmind/code_contests", data_dir='data/', split='valid')

    # If the dataset is gated/private, make sure you have run huggingface-cli login
    dataset = load_dataset("nampdn-ai/tiny-codes")
    #dataset = load_dataset("iamtarun/code_instructions_120k_alpaca")
    return dataset

def extract_keywords_from_dataset(dataset):
    keywords = set()
    for split in dataset.keys():
        for item in dataset[split]:
            if 'keywords' in item:
                keywords.update(item['keywords'])
    return keywords

def save_keywords_to_file(keywords, filename="keywords.json"):
    with open(filename, 'w') as f:
        json.dump(list(keywords), f, indent=4)

if __name__ == "__main__":
    dataset = download_xcode_eval_dataset()
    print(dataset)
    # keywords = extract_keywords_from_dataset(dataset)
    # save_keywords_to_file(keywords)
    # print(f"Extracted {len(keywords)} unique keywords from the xCodeEval dataset.")
