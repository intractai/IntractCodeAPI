import copy
from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any, List, Optional, Dict, Sequence
import warnings

from datasets import load_dataset, Dataset
import numpy as np
import torch
import torch.distributed
import transformers
from transformers import Trainer

from src import modeling


# Labels with -100 index are ignored in the loss calculation
# HuggingFace... document this stuff please
IGNORE_INDEX = -100

EOT_TOKEN = "<|EOT|>"
FIM_BEGIN_TOKEN = "<｜fim▁begin｜>"
FIM_HOLE_TOKEN = "<｜fim▁hole｜>"
FIM_END_TOKEN = "<｜fim▁end｜>"

EOT_ID = None
FIM_BEGIN_ID = None
FIM_HOLE_ID = None
FIM_END_ID = None

CHUNK_OVERLAP_FRAC = 0.1 # Fraction of chunk to overlap with previous chunk
                         # When splitting a document into chunks for finetuning
RESERVED_TOKENS = 32 # Number of tokens reserved for file path and code insertion tokens
FIM_RATE = 1 # Probability of performing FIM on a code chunk


def build_projectcode_prompt(data: dict):
    data_str = ""
    for file_path, file_code in data.items():
        data_str += f"# {file_path}\n{file_code}"
    return data_str


def prepare_raw_dataset(project_data: Dict[str, str]) -> Dataset:
    """Converts a project dictionary into HF dataset."""

    data = []
    for file_path, file_code in project_data.items():
        data.append({'file_path': file_path, 'code': file_code})
    return Dataset.from_list(data)


def prepare_code_files(elements: Dict[str, Any], tokenizer, max_length=None) -> Dict[str, Any]:
    """Chunk and tokenize code and corresponding file path."""

    max_length = max_length or tokenizer.model_max_length

    # Tokenize the code
    code_tokens = tokenizer(
            elements['code'],
            max_length=max_length,
            truncation=True,
            return_overflowing_tokens=True,
            stride=int(max_length * CHUNK_OVERLAP_FRAC), # Amount of overlap between chunks
            add_special_tokens=False,
        )

    # Tokenize the file paths
    file_path_text = [f"# {fp}\n" for fp in elements['file_path']]
    file_path_tokens = tokenizer(
            file_path_text,
            max_length=RESERVED_TOKENS,
            truncation=True,
            add_special_tokens=False,
        )

    # Repeat some file path tokens for any overflow code chunks
    extended_fp_tokens = []
    sample_map = code_tokens.pop('overflow_to_sample_mapping')
    for sample_id in sample_map:
        extended_fp_tokens.append(file_path_tokens['input_ids'][sample_id])

    return dict(
        code_tokens=code_tokens.input_ids,
        file_path_tokens=extended_fp_tokens
    )


# Adapted from https://github.com/EleutherAI/gpt-neox/blob/
# FIM-clean/megatron/data/gpt2_dataset.py#L339
def make_fim_example(code_tokens, fp_tokens, tokenizer, truncate=True):
    """
    Take in a sample (np array w/ size (0,chunklength)) and perform a FIM transformation on it. 
    Maintain the same sample length (if transform creates a few extra tokens, drop them).
    """

    contents = tokenizer.decode(code_tokens, skip_special_tokens=False)

    try:
        # A boundary can be = 0 (prefix will be empty)
        # a boundary can be = len(contents) (suffix will be empty)
        # The two boundaries can be equal (middle will be empty)
        boundaries = list(np.random.randint(low=0, high=len(contents) + 1, size=2))
        boundaries.sort()
    except ValueError as e:
        print(len(contents), contents)
        print(e)
        raise e

    prefix = contents[:boundaries[0]]
    middle = contents[boundaries[0]:boundaries[1]]
    suffix = contents[boundaries[1]:]

    prefix = np.array(tokenizer.encode(prefix, add_special_tokens=False), dtype=np.int64)
    middle = np.array(tokenizer.encode(middle, add_special_tokens=False), dtype=np.int64)
    suffix = np.array(tokenizer.encode(suffix, add_special_tokens=False), dtype=np.int64)

    prefix_tok_id = tokenizer.vocab[FIM_BEGIN_TOKEN]
    middle_tok_id = tokenizer.vocab[FIM_HOLE_TOKEN]
    suffix_tok_id = tokenizer.vocab[FIM_END_TOKEN]

    # Here we truncate each given segment to fit the same length as it was before
    # A consequence is that we never reach the end of a file?
    # we should rather truncate at the context-level
    if truncate:
        # need to make same length as the input. Take the 3 sentinel tokens into account
        new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0] + 3
        diff = new_length - tokenizer.model_max_length
        if diff > 0: # too long
            # If there's no space to truncate the suffix:
            # stop and report it. atm i should have stopped this from happening
            if suffix.shape[0] <= diff:
                return None
            suffix = suffix[:suffix.shape[0] - diff]

    input_ids = np.concatenate([
        [prefix_tok_id], fp_tokens, prefix,
        [middle_tok_id], suffix,
        [suffix_tok_id], middle
    ])

    # Ignore the prompt tokens when calculating loss
    labels = np.concatenate([
        [IGNORE_INDEX] * (len(fp_tokens) + len(prefix) + 1),
        [IGNORE_INDEX] * (len(suffix) + 1),
        [IGNORE_INDEX], middle
    ])

    return dict(
        input_ids=input_ids,
        labels=labels
    )


def prepare_train_dataset(elements: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """Convert code and file path tokens into training data"""

    input_ids = []
    labels = []

    # First generate next token prediction training data

    for i in range(len(elements['code_tokens'])):
        fp_tokens = elements['file_path_tokens'][i]
        code_tokens = elements['code_tokens'][i]
        input_ids.append(
            [tokenizer.bos_token_id] + \
            fp_tokens + \
            code_tokens + \
            [tokenizer.eos_token_id]
        )
        labels.append(
            [IGNORE_INDEX] + \
            len(fp_tokens) * [IGNORE_INDEX] + \
            code_tokens + \
            [IGNORE_INDEX]
        )

    # Then generate fill in the blank training data

    for i in range(len(elements['code_tokens'])):
        if np.random.binomial(1, FIM_RATE):
            fim_sample = make_fim_example(code_tokens, fp_tokens, tokenizer)
            if fim_sample is None:
                warnings.warn("FIM sample was too long, skipping")
            else:
                input_ids.append(fim_sample['input_ids'])
                labels.append(fim_sample['labels'])

    return dict(
        input_ids=input_ids,
        labels=labels
    )


def project_to_dataset(
        project_data: Dict[str, str],
        tokenizer,) -> Dataset:
    """Converts a project dictionary into HF dataset."""

    raw_dataset = prepare_raw_dataset(project_data)

    chunk_size = tokenizer.model_max_length - RESERVED_TOKENS
    code_dataset = raw_dataset.map(
        prepare_code_files,
        batched=True,
        batch_size=3000,
        num_proc=1,
        load_from_cache_file=True,  # not args.overwrite_cache
        remove_columns=raw_dataset.column_names,
        fn_kwargs={'tokenizer': tokenizer, 'max_length': chunk_size}
    )
    train_dataset = code_dataset.map(
        prepare_train_dataset,
        batched=True,
        batch_size=3000,
        num_proc=1,
        load_from_cache_file=True,  # not args.overwrite_cache
        remove_columns=code_dataset.column_names,
        fn_kwargs={'tokenizer': tokenizer}
    )

    return train_dataset


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    # Number of code insertion prompts to generate per file
    # num_code_insertions_per_file: int = 10
    # span_max: int = 256  # Range of spans to insert code into, span is randomly selected
    #                      # from a poisson distribution with mean 1


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def train_supervised_projectdir(project_data, eval_data=None, **kwargs):
    # Eval data must be in format {name_of_dataset: {file_name: file_contents, ...}},
    # even if only one eval dataset
    # ModelArguments, DataArguments, TrainingArguments
    # torch.set_num_threads(1)

    parser = transformers.HfArgumentParser((TrainingArguments))
    training_args = parser.parse_dict(kwargs)[0]
    training_args.do_train = True

    if training_args.local_rank == 0:
        print('=' * 100)
        print(training_args)

    tokenizer = modeling.GLOBAL_TOKENIZER

    print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

    # Store reference to the model and tokenizer in the model module
    model = modeling.GLOBAL_MODEL

    # assert model

    train_dataset = project_to_dataset(project_data, tokenizer)

    if eval_data is None:
        eval_datasets = None
    else:
        eval_datasets = {}
        for dataset_name, data in eval_data.items():
            eval_datasets[dataset_name] = project_to_dataset(data, tokenizer)

        if len(eval_datasets) == 1:
            eval_datasets = eval_datasets[list(eval_datasets.keys())[0]]

    # if training_args.local_rank == 0:
    #     torch.distributed.barrier()

    # if training_args.local_rank == 0:
    #     print("Training dataset samples:", len(train_dataset))
    #     # for index in random.sample(range(len(train_dataset)), 3):
    #     for index in range(3):  # Print the first 3 samples
    #         print(train_dataset[index]['input_ids'])
    #         print(f"Sample {index} of the training set: {train_dataset[index]['input_ids']}," +
    #               f"{train_dataset[index]['labels']}.")
    #         print(f"Sample {index} of the training set:" +
    #               f"{tokenizer.decode(list(train_dataset[index]['input_ids']))}.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset,
                       eval_dataset=eval_datasets, data_collator=data_collator)

    trainer = Trainer(model=model, tokenizer=tokenizer,
                      args=training_args, **data_module)

    trainer.train()
    
    # Release all the memory used by the trainer
    trainer.model = None
    trainer.tokenizer = None
    trainer.optimizer = None
    trainer.lr_scheduler = None
    trainer.state = None
    trainer.args = None
    torch.cuda.empty_cache()

    # trainer.save_state()

    # modeling.GLOBAL_MODEL = trainer.model
