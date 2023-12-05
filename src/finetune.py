import copy
from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Optional, Dict, Sequence

from datasets import load_dataset, Dataset
import numpy as np
import torch
import torch.distributed
import transformers
from transformers import Trainer

from src import modeling


IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"
FIM_BEGIN_TOKEN = "<｜fim▁begin｜>"
FIM_HOLE_TOKEN = "<｜fim▁hole｜>"
FIM_END_TOKEN = "<｜fim▁end｜>"


def build_projectcode_prompt(data: dict):
    data_str = ""
    for file_path, file_code in data.items():
        data_str += f"#{file_path}\n{file_code}\n\n\n"
    return data_str


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    # Number of code insertion prompts to generate per file
    num_code_insertions_per_file: int = 10
    span_max: int = 256  # Range of spans to insert code into, span is randomly selected from a poisson distribution with mean 1


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu()
                          for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0]
                          for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(
        strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


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


def train_tokenize_function(examples, tokenizer):
    sources = examples['source']
    targets = examples['target']
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict


def generate_dataset_from_projectdir(project_data, num_code_insertions, span_max):
    dataset = []
    # First element in the Dataset is next token prediction on the project data
    project_data_source = build_projectcode_prompt(project_data)
    dataset.append({'source': "", "target": project_data_source})
    # Rest of the elements are fill in the blank prompts
    for file_path, file_code in project_data.items():
        for _ in range(num_code_insertions):
            file_code_per_line = file_code.split("\n")
            span_start = random.randint(0, len(file_code_per_line) - 1)
            span_length = np.clip(np.random.poisson(1), 1, span_max)
            span_end = min(span_start + span_length, len(file_code_per_line))
            code_begin = "\n".join(file_code_per_line[:span_start])
            code_hole = "\n".join(file_code_per_line[span_start:span_end])
            code_end = "\n".join(file_code_per_line[span_end:])
            source = f"{FIM_BEGIN_TOKEN}{code_begin}\n{FIM_HOLE_TOKEN}\n{code_end}{FIM_END_TOKEN}"
            target = code_hole
            dataset.append({'source': source, "target": target})
    return Dataset.from_list(dataset)


def train_supervised_projectdir(project_data, eval_dataset=None, **kwargs):
    # ModelArguments, DataArguments, TrainingArguments
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

    dataset = generate_dataset_from_projectdir(
        project_data, training_args.num_code_insertions_per_file,
        training_args.span_max)

    train_dataset = dataset.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        load_from_cache_file=True,  # not args.overwrite_cache
        desc="Running Encoding",
        fn_kwargs={"tokenizer": tokenizer}
    )

    # if training_args.local_rank == 0:
    #     torch.distributed.barrier()

    if training_args.local_rank == 0:
        print("Training dataset samples:", len(train_dataset))
        # for index in random.sample(range(len(train_dataset)), 3):
        for index in range(3):  # Print the first 3 samples
            print(train_dataset[index]['input_ids'])
            print(f"Sample {index} of the training set: {train_dataset[index]['input_ids']}," +
                  f"{train_dataset[index]['labels']}.")
            print(f"Sample {index} of the training set:" +
                  f"{tokenizer.decode(list(train_dataset[index]['input_ids']))}.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset,
                       eval_dataset=eval_dataset, data_collator=data_collator)

    trainer = Trainer(model=model, tokenizer=tokenizer,
                      args=training_args, **data_module)

    trainer.train()
    trainer.save_state()
    # safe_save_model_for_hf_trainer(
    #     trainer=trainer, output_dir=training_args.output_dir)
    modeling.GLOBAL_MODEL = trainer.model
