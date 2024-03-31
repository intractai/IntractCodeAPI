from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence
import warnings

from datasets import Dataset
import numpy as np
from omegaconf import DictConfig
import torch
import torch.distributed
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.training.data_formatting import (
    prepare_fim_train_input,
    prepare_ntp_train_input,
    MAX_FP_TOKENS
)
from src.training.trainer import ContinualTrainer


logger = logging.getLogger(__name__)


# Labels with -100 index are ignored in the loss calculation
# HuggingFace... document this stuff please
IGNORE_INDEX = -100
CHUNK_OVERLAP_FRAC = 0.1 # Fraction of chunk to overlap with previous chunk
                         # When splitting a document into chunks for finetuning
FIM_RATE = 1 # Probability of performing FIM on a code chunk
DEFAULT_TRAIN_METHODS = ['ntp']


def prepare_raw_project_dataset(project_data: Dict[str, str]) -> Dataset:
    """Converts a project dictionary into HF dataset."""

    data = []
    for file_path, file_code in project_data.items():
        data.append({'file_path': file_path, 'code': file_code})
    return Dataset.from_list(data)


def prepare_raw_document_dataset(documents: List[str]) -> Dataset:
    """Converts a list of documents into HF dataset."""
    return Dataset.from_list([{'text': x} for x in documents])


def prepare_code_files(elements: Dict[str, Any], tokenizer, max_length=None) -> Dict[str, Any]:
    """Chunk and tokenize code and corresponding file path."""

    max_length = max_length or tokenizer.model_max_length

    # Tokenize the code
    code_tokens = tokenizer(
        elements['code'],
        max_length = max_length,
        truncation = True,
        return_overflowing_tokens = True,
        stride = int(max_length * CHUNK_OVERLAP_FRAC), # Amount of overlap between chunks
        add_special_tokens = False,
    )

    # Tokenize the file paths
    file_path_text = [f"# {fp}\n" for fp in elements['file_path']]
    file_path_tokens = tokenizer(
        file_path_text,
        max_length = MAX_FP_TOKENS,
        truncation = True,
        add_special_tokens = False,
    )

    # Repeat some file path tokens for any overflow code chunks
    extended_fp_tokens = []
    sample_map = code_tokens.pop('overflow_to_sample_mapping')
    for sample_id in sample_map:
        extended_fp_tokens.append(file_path_tokens['input_ids'][sample_id])

    return dict(
        code_tokens = code_tokens.input_ids,
        file_path_tokens = extended_fp_tokens
    )


def tokenize_documents(elements: Dict[str, Any], tokenizer, max_length=None) -> Dict[str, Any]:
    """Chunk and tokenize documents."""

    max_length = max_length or tokenizer.model_max_length

    # TODO: Consider adding single bos/eos tokens before chunking
    modified_text = [t for t in elements['text']]

    # Tokenize the code
    tokenized_text = tokenizer(
        modified_text,
        max_length=max_length,
        truncation=True,
        return_overflowing_tokens=True,
        stride=int(max_length * CHUNK_OVERLAP_FRAC), # Amount of overlap between chunks
        add_special_tokens=False,
    )
    
    return dict(input_ids=tokenized_text.input_ids)


def prepare_projects_train_dataset(
        elements: Dict[str, Any], tokenizer: transformers.PreTrainedTokenizer,
        train_methods: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
    """Convert code and file path tokens into training data"""
    train_methods = train_methods or DEFAULT_TRAIN_METHODS

    input_ids = []
    labels = []
    sample_types = []

    # First generate next token prediction training data
    if 'ntp' in train_methods:
        for i in range(len(elements['code_tokens'])):
            code_tokens = elements['code_tokens'][i]
            fp_tokens = elements['file_path_tokens'][i]

            sample = prepare_ntp_train_input(code_tokens, fp_tokens, tokenizer)

            input_ids.append(sample['input_ids'])
            labels.append(sample['labels'])
            sample_types.append('ntp')

    # Then generate fill in the blank training data
    if 'fim' in train_methods:
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
        input_ids = input_ids,
        labels = labels,
        sample_types = sample_types
    )


def prepare_documents_train_dataset(
        elements: Dict[str, Any], tokenizer: transformers.PreTrainedTokenizer,
        train_methods: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
    """Convert tokenized text into training data."""
    train_methods = train_methods or DEFAULT_TRAIN_METHODS

    input_ids = []
    labels = []
    sample_types = []

    # First generate next token prediction training data

    if 'ntp' in train_methods:
        for tokens in elements['input_ids']:
            sample = prepare_ntp_train_input(tokens, [], tokenizer)
            input_ids.append(sample['input_ids'])
            labels.append(sample['labels'])
            sample_types.append('ntp')

    # Then generate fill in the blank training data
    if 'fim' in train_methods:
        for tokens in elements['input_ids']:
            if FIM_RATE > 1:
                do_extra_fim = np.random.binomial(1, FIM_RATE - int(FIM_RATE))
                n_samples = int(FIM_RATE) + do_extra_fim
            else:
                n_samples = np.random.binomial(1, FIM_RATE)

            for _ in range(n_samples):
                fim_sample = prepare_fim_train_input(tokens, [], tokenizer)
                if fim_sample is None:
                    logger.warn("FIM sample was too long, skipping")
                else:
                    input_ids.append(fim_sample['input_ids'])
                    labels.append(fim_sample['labels'])
                    sample_types.append('fim')

    return dict(
        input_ids = input_ids,
        labels = labels,
        sample_types = sample_types
    )


def project_to_dataset(
        project_data: Dict[str, str],
        tokenizer: transformers.PreTrainedTokenizer,
        train_methods: Optional[List[str]] = None,
    ) -> Dataset:
    """Converts a project dictionary into HF dataset."""

    raw_dataset = prepare_raw_project_dataset(project_data)

    # -1 for for BOS token
    chunk_size = tokenizer.model_max_length - MAX_FP_TOKENS - 1
    code_dataset = raw_dataset.map(
        prepare_code_files,
        batched = True,
        batch_size = 3000,
        num_proc = 1,
        load_from_cache_file = True,  # not args.overwrite_cache
        remove_columns = raw_dataset.column_names,
        fn_kwargs = {'tokenizer': tokenizer, 'max_length': chunk_size},
    )
    train_dataset = code_dataset.map(
        prepare_projects_train_dataset,
        batched = True,
        batch_size = 3000,
        num_proc = 1,
        load_from_cache_file = True,  # not args.overwrite_cache
        remove_columns = code_dataset.column_names,
        fn_kwargs = {'tokenizer': tokenizer, 'train_methods': train_methods},
    )

    return train_dataset


def documents_to_dataset(
        documents: List[str],
        tokenizer: transformers.PreTrainedTokenizer,
        train_methods: Optional[List[str]] = None,
    ) -> Dataset:
    """Converts a project dictionary into HF dataset."""

    raw_dataset = prepare_raw_document_dataset(documents)

    # -1 for for BOS token
    chunk_size = tokenizer.model_max_length - MAX_FP_TOKENS - 1
    code_dataset = raw_dataset.map(
        tokenize_documents,
        batched = True,
        batch_size = 3000,
        num_proc = 1,
        load_from_cache_file = True,  # not args.overwrite_cache
        remove_columns = raw_dataset.column_names,
        fn_kwargs = {'tokenizer': tokenizer, 'max_length': chunk_size},
    )
    train_dataset = code_dataset.map(
        prepare_documents_train_dataset,
        batched = True,
        batch_size = 3000,
        num_proc = 1,
        load_from_cache_file = True,  # not args.overwrite_cache
        # remove_columns=code_dataset.column_names,
        fn_kwargs = {'tokenizer': tokenizer, 'train_methods': train_methods},
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


def train_self_supervised(
        model: PreTrainedModel, tokenizer: PreTrainedTokenizer, train_dataset: Dataset,
        config: DictConfig, eval_datasets: Optional[Dict[str, Dataset]] = None,
        compute_metrics = None, metrics_collator: Callable = None, **kwargs,
    ):
    """Train a model in a self-supervised manner given a dataset.

    This function serves as the backbone for the other training functions in this module.
    The dataset is expected to be fully formatted and ready for training.
    
    Args:
        model (PreTrainedModel): The model to train.
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
        train_dataset (Dataset): The dataset to train on.
        eval_datasets (Dict[str, Dataset]): The datasets to evaluate on.
        compute_metrics (Callable): The function to compute metrics.
        metrics_collator (Callable): The function to collate metrics.
        config (dict): The training configuration.
        **kwargs: Additional arguments passed to trainer args.
    """
    parser = transformers.HfArgumentParser((TrainingArguments))
    training_args = parser.parse_dict({**config.trainer, **kwargs})[0]

    if training_args.local_rank == 0:
        logger.debug("Train args:\n" + str(training_args))

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    trainer = ContinualTrainer(
        model = model,
        tokenizer = tokenizer,
        args = training_args,
        compute_metrics = compute_metrics,
        metrics_collator = metrics_collator,
        train_dataset = train_dataset,
        eval_dataset = eval_datasets,
        data_collator = data_collator,
    )

    trainer.train()

    # Release all the memory used by the trainer
    # Had some memory leak issues and this maybe solved it?
    trainer.model = None
    trainer.tokenizer = None
    trainer.optimizer = None
    trainer.lr_scheduler = None
    trainer.state = None
    trainer.args = None
    del trainer

    torch.cuda.empty_cache()


def train_self_supervised_documents(
        model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
        documents: List[str], config: DictConfig, eval_data: Optional[List[str]] = None,
        compute_metrics = None, metrics_collator: Callable = None,
        train_methods: Optional[List[str]] = None, **kwargs,
    ):
    """Train a model in a self-supervised manner given a list of strings.

    For arguments other than documents, eval_data, and train_methods, see train_self_supervised.

    Args:
        documents (List[str]): The documents to train on.
        eval_data (List[str]): The evaluation data.
        train_methods (List[str]): The training methods to use.
            'ntp' (next token prediction) and 'fim' (fill-in-the-middle) are available.
            If None, only ntp is used.
    """
    train_dataset = documents_to_dataset(documents, tokenizer, train_methods)

    if eval_data is None:
        eval_datasets = None
    else:
        eval_datasets = {}
        for dataset_name, data in eval_data.items():
            eval_datasets[dataset_name] = documents_to_dataset(data, tokenizer, train_methods)

    train_self_supervised(
        model, tokenizer, train_dataset, config, eval_datasets,
        compute_metrics, metrics_collator, **kwargs
    )


def train_self_supervised_project(
        model: PreTrainedModel, tokenizer: PreTrainedTokenizer, project_data,
        config: DictConfig, eval_data = None, compute_metrics = None,
        metrics_collator: Callable = None, train_methods: Optional[List[str]] = None,
        **kwargs,
    ):
    """Train a model in a self-supervised manner given a dataset.
    
    For arguments other than project_data, eval_data, and train_methods,
    see train_self_supervised.

    Args:
        project_data (Dict[str, str]): The project data to train on.
        eval_data (Dict[str, Dict[str, str]]): The evaluation data.
        train_methods (List[str]): The training methods to use.
            'ntp' (next token prediction) and 'fim' (fill-in-the-middle) are available.
            If None, only ntp is used.
    """
    # Eval data must be in format {name_of_dataset: {file_name: file_contents, ...}},
    # even if only one eval dataset
    train_dataset = project_to_dataset(project_data, tokenizer, train_methods)

    if eval_data is None:
        eval_datasets = None
    else:
        eval_datasets = {}
        for dataset_name, data in eval_data.items():
            eval_datasets[dataset_name] = project_to_dataset(data, tokenizer, train_methods)

    train_self_supervised(
        model, tokenizer, train_dataset, config, eval_datasets,
        compute_metrics, metrics_collator, **kwargs
    )