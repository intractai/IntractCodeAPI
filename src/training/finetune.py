from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence
import warnings

from datasets import Dataset
import numpy as np
import torch
import torch.distributed
import transformers

from src import modeling
from src.data_formatting import (
    prepare_fim_train_input,
    prepare_ntp_train_input,
    MAX_FP_TOKENS
)
from src.trainer import ContinualTrainer


# Labels with -100 index are ignored in the loss calculation
# HuggingFace... document this stuff please
IGNORE_INDEX = -100

CHUNK_OVERLAP_FRAC = 0.1 # Fraction of chunk to overlap with previous chunk
                         # When splitting a document into chunks for finetuning
FIM_RATE = 1 # Probability of performing FIM on a code chunk


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
            max_length=MAX_FP_TOKENS,
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


def prepare_train_dataset(elements: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """Convert code and file path tokens into training data"""

    input_ids = []
    labels = []
    sample_types = []

    # First generate next token prediction training data

    for i in range(len(elements['code_tokens'])):
        code_tokens = elements['code_tokens'][i]
        fp_tokens = elements['file_path_tokens'][i]

        sample = prepare_ntp_train_input(code_tokens, fp_tokens, tokenizer)

        input_ids.append(sample['input_ids'])
        labels.append(sample['labels'])
        sample_types.append('ntp')

    # Then generate fill in the blank training data

    for i in range(len(elements['code_tokens'])):
        if FIM_RATE > 1:
            do_extra_fim = np.random.binomial(1, FIM_RATE - int(FIM_RATE))
            n_samples = int(FIM_RATE) + do_extra_fim
        else:
            n_samples = np.random.binomial(1, FIM_RATE)

        for _ in range(n_samples):
            fim_sample = prepare_fim_train_input(code_tokens, fp_tokens, tokenizer)
            if fim_sample is None:
                warnings.warn("FIM sample was too long, skipping")
            else:
                input_ids.append(fim_sample['input_ids'])
                labels.append(fim_sample['labels'])
                sample_types.append('fim')

    return dict(
        input_ids=input_ids,
        labels=labels,
        sample_types=sample_types
    )


def project_to_dataset(
        project_data: Dict[str, str],
        tokenizer,) -> Dataset:
    """Converts a project dictionary into HF dataset."""

    raw_dataset = prepare_raw_dataset(project_data)

    # -1 for for BOS token
    chunk_size = tokenizer.model_max_length - MAX_FP_TOKENS - 1
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


def train_supervised_projectdir(
        project_data, eval_data=None, compute_metrics=None,
        metrics_collator=None, train_cfg: dict = None, **kwargs):
    # Eval data must be in format {name_of_dataset: {file_name: file_contents, ...}},
    # even if only one eval dataset
    # ModelArguments, DataArguments, TrainingArguments
    # torch.set_num_threads(1)

    model_provider = modeling.ModelProvider.get_instance()

    parser = transformers.HfArgumentParser((TrainingArguments))
    training_args = parser.parse_dict({**train_cfg, **kwargs})[0]

    if training_args.local_rank == 0:
        print('=' * 100)
        print(training_args)

    tokenizer = model_provider.get_model_utils()['tokenizer']

    print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

     # Store reference to the model and tokenizer in the model module
    model = model_provider.get_model()

    train_dataset = project_to_dataset(project_data, tokenizer)

    if eval_data is None:
        eval_datasets = None
    else:
        eval_datasets = {}
        for dataset_name, data in eval_data.items():
            eval_datasets[dataset_name] = project_to_dataset(data, tokenizer)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset,
                       eval_dataset=eval_datasets, data_collator=data_collator)

    trainer = ContinualTrainer(
        model = model,
        tokenizer = tokenizer,
        args = training_args,
        compute_metrics = compute_metrics,
        metrics_collator = metrics_collator,
        **data_module
    )

    trainer.train()
    model_provider.update_model(trainer.model)

    # Release all the memory used by the trainer
    trainer.model = None
    trainer.tokenizer = None
    trainer.optimizer = None
    trainer.lr_scheduler = None
    trainer.state = None
    trainer.args = None
    torch.cuda.empty_cache()

    # trainer.save_state()
