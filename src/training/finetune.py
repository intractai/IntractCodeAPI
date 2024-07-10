from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence
import warnings

from datasets import concatenate_datasets, Dataset
import numpy as np
from omegaconf import DictConfig
import torch
import torch.distributed
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.training.data_formatting import (
    FIMTrainSample,
    MAX_FP_TOKENS,
    prepare_fim_train_input,
    prepare_ntp_train_input,
    text_to_fim_train_sample,
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
COMMON_DATASET_MAP_KWARGS = {
    'batched': True,
    'batch_size': 3000,
    'num_proc': 1,
    'load_from_cache_file': True,
}


def prepare_raw_project_dataset(project_data: Dict[str, str]) -> Dataset:
    """Converts a project dictionary into HF dataset."""

    data = []
    for file_path, file_code in project_data.items():
        data.append({'file_path': file_path, 'code': file_code})
    return Dataset.from_list(data)


def prepare_raw_document_dataset(documents: List[str]) -> Dataset:
    """Converts a list of documents into HF dataset."""
    return Dataset.from_list([{'text': x} for x in documents])


def chunk_files(elements: Dict[str, Any], tokenizer, tokenize=False, max_length=None) -> Dict[str, Any]:
    """Chunk and tokenize code and corresponding file path."""

    max_length = max_length or tokenizer.model_max_length

    # Tokenize the code
    code_results = tokenizer(
        elements['code'],
        max_length = max_length,
        truncation = True,
        return_overflowing_tokens = True,
        stride = int(max_length * CHUNK_OVERLAP_FRAC), # Amount of overlap between chunks
        add_special_tokens = False,
        return_offsets_mapping = True,
    )
    code_text = []
    for i, sample_idx in enumerate(code_results.overflow_to_sample_mapping):
        start_idx = code_results.offset_mapping[i][0][0]
        end_idx = code_results.offset_mapping[i][-1][1]
        code_text.append(elements['code'][sample_idx][start_idx:end_idx])

    # Tokenize the file paths
    file_path_results = tokenizer(
        elements['file_path'],
        max_length = max(MAX_FP_TOKENS - 1, 0),
        truncation = True,
        add_special_tokens = False,
    )

    # Repeat some file path tokens for any overflow code chunks
    extended_fp_tokens = []
    extended_fp_text = []
    sample_map = code_results.overflow_to_sample_mapping
    for sample_id in sample_map:
        extended_fp_tokens.append(file_path_results.input_ids[sample_id])
        extended_fp_text.append(elements['file_path'][sample_id])

    return_dict = dict(
        code_text = code_text,
        file_path_text = extended_fp_text,
    )

    if tokenize:
        return_dict['code_tokens'] = code_results.input_ids
        return_dict['file_path_tokens'] = extended_fp_tokens

    return return_dict


def convert_to_inference_format(
        elements: Dict[str, Any],
        code_key: str = 'code_text',
        file_path_key = 'file_path_text',
    ) -> Dict[str, Any]:
    """Converts the chunked data into a format suitable for inference.
    
    Matches the key format used during inference:
        prior_context: str
        file_path: Optional[str] = None
        proceeding_context: Optional[str] = None

    target_text and sample_type are also added for training-specific transformations.
    target_text is used later for FIM samples.
    sample_type is used to filter out samples of unwanted training methods if necessary.
    """
    return dict(
        prior_context = elements[code_key],
        file_path = elements[file_path_key] if file_path_key else None,
        proceeding_context = [None for _ in range(len(elements[code_key]))],
        target_text = ['' for _ in range(len(elements[code_key]))],
        sample_type = ['ntp' for _ in range(len(elements[code_key]))],
    )


def generate_fim_samples(elements: Dict[str, Any]) -> Dict[str, Any]:
    """Converts a number of ntp examples into fim examples.

    Returns only the new fim samples, discarding all of the original ntp samples.

    Args:
        elements (Dict[str, Any]): The dataset elements to convert.
            Should contain columns 'prior_context', 'file_path', 'proceeding_context',
            'target_text', and 'sample_type'.
    """
    fim_samples = {}
    for key in ['prior_context', 'proceeding_context', 'target_text', 'file_path', 'sample_type']:
        fim_samples[key] = []
        if key not in elements:
            raise ValueError(f"Requires key '{key}' in elements!")

    for i in range(len(elements['prior_context'])):
        # Determine how many FIM samples to generate for this sample
        if FIM_RATE > 1:
            do_extra_fim = np.random.binomial(1, FIM_RATE - int(FIM_RATE))
            n_samples = int(FIM_RATE) + do_extra_fim
        else:
            n_samples = np.random.binomial(1, FIM_RATE)

        for _ in range(n_samples):
            fim_sample = text_to_fim_train_sample(elements['prior_context'][i])
            fim_samples['prior_context'].append(fim_sample.preceeding_context)
            fim_samples['proceeding_context'].append(fim_sample.proceeding_context)
            fim_samples['target_text'].append(fim_sample.target_text)
            fim_samples['file_path'].append(elements['file_path'][i])
            fim_samples['sample_type'].append('fim')

    return fim_samples


def project_to_problems(
        project_data: Dict[str, str],
        tokenizer: transformers.PreTrainedTokenizer,
        train_methods: Optional[List[str]] = None,
        context_size: Optional[int] = None,
    ) -> Dataset:
    """Converts a project dictionary into HF dataset of problems.
    
    Args:
        project_data (Dict[str, str]): Dict mapping from file path to text (usually code).
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used for chunking large files.
        train_methods (List[str]): The training methods to use.
            'ntp' (next token prediction) and 'fim' (fill-in-the-middle) are available.
            If None, only ntp is used.
        context_size (int): The maximum number of tokens per problem, which is used to determine
            the size of chunks. If None, the models maximum context size (specified in the config) is used.
    """
    raw_dataset = prepare_raw_project_dataset(project_data)

    # -1 for for BOS token
    chunk_size = context_size or tokenizer.model_max_length
    chunk_size -= MAX_FP_TOKENS + 1

    # First chunk the dataset into subsets of text
    # Each chunk separately stores the text and the file path
    train_problems = raw_dataset.map(
        chunk_files,
        remove_columns = raw_dataset.column_names,
        fn_kwargs = {'tokenizer': tokenizer, 'max_length': chunk_size, 'tokenize': False},
        **COMMON_DATASET_MAP_KWARGS,
    )

    # Then change the keys to match the format given for inference (prior and proceeding context)
    # We split it into columns for prior_context, proceeding_context, target_text, file_path, sample_type
    # NTP samples store everything in prior_context, but FIM samples also use the proceeding and target text columns
    train_problems = train_problems.map(
        convert_to_inference_format,
        remove_columns = train_problems.column_names,
        fn_kwargs = {'code_key': 'code_text', 'file_path_key': 'file_path_text'},
        **COMMON_DATASET_MAP_KWARGS,
    )

    # Now we use the NTP samples to generate FIM samples by removing random chunks of code
    if 'fim' in train_methods:
        fim_dataset = train_problems.map(
            generate_fim_samples,
            **COMMON_DATASET_MAP_KWARGS,
        )

    # If we only wanted to generate FIM samples (indicated by `train_methods`), then we can discard the NTP samples
    if 'ntp' not in train_methods:
        train_problems = fim_dataset
    # Otherwise we can combined the two datasets
    else:
        train_problems = concatenate_datasets([train_problems, fim_dataset])

    return train_problems


def problems_to_train_samples(
        elements: Dict[str, Any],
        tokenizer: transformers.PreTrainedTokenizer,
        additional_context_max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
    """Converts train problems to samples formatted for training.
    
    Args:
        elements (Dict[str, Any]): The dataset elements to convert.
            Should contain columns 'prior_context', 'proceeding_context', 'target_text',
            'file_path', 'sample_type', and optionally, 'additional_context'.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.

    Returns:
        Dict[str, Any]: The formatted training samples containing input_ids, labels, and sample_types.
    """
    input_ids = []
    labels = []
    sample_types = []

    file_path_tokens = tokenizer([f'# {fp}' for fp in elements['file_path']], add_special_tokens=False).input_ids
    code_tokens = tokenizer(elements['prior_context'], add_special_tokens=False).input_ids

    # Format and tokenize additional context (usually from RAG)
    if 'additional_context' in elements:
        truncate = additional_context_max_length is not None
        additional_context_tokens = tokenizer(
            elements['additional_context'], add_special_tokens=False,
            truncation=truncate, max_length=additional_context_max_length,
        ).input_ids
    else:
        additional_context_tokens = [[] for _ in elements['prior_context']]

    for i in range(len(elements['sample_type'])):

        # Handle next token prediction formatting
        if elements['sample_type'][i] == 'ntp':
            sample = prepare_ntp_train_input(
                code_tokens[i], file_path_tokens[i], tokenizer,
                additional_context_tokens=additional_context_tokens[i],
            )

            input_ids.append(sample['input_ids'])
            labels.append(sample['labels'])
            sample_types.append('ntp')

        # Handle fill-in-the-middle formatting
        elif elements['sample_type'][i] == 'fim':
            fim_data = FIMTrainSample(
                elements['prior_context'][i], elements['proceeding_context'][i], elements['target_text'][i])
            fim_sample = prepare_fim_train_input(
                tokenizer, file_path_tokens[i], fim_sample=fim_data,
                additional_context_tokens=additional_context_tokens[i],
            )

            if fim_sample is None:
                warnings.warn("FIM sample was too long, skipping")
            else:
                input_ids.append(fim_sample['input_ids'])
                labels.append(fim_sample['labels'])
                sample_types.append('fim')

    return dict(
        input_ids = input_ids,
        labels = labels,
        sample_types = sample_types,
    )


def project_to_train_dataset(
        project_data: Dict[str, str],
        tokenizer: transformers.PreTrainedTokenizer,
        train_methods: Optional[List[str]] = None,
        context_size: Optional[int] = None,
    ) -> Dataset:
    """Converts a project dictionary into HF dataset of train samples."""

    # Convert the project into a dataset of problems
    problems_dataset = project_to_problems(project_data, tokenizer, train_methods, context_size)

    # Convert those into train samples by tokenizing and formatting them for training
    train_dataset = problems_dataset.map(
        problems_to_train_samples,
        fn_kwargs = {'tokenizer': tokenizer},
        remove_columns = problems_dataset.column_names,
        **COMMON_DATASET_MAP_KWARGS,
    )

    return train_dataset


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
                fim_sample = prepare_fim_train_input(tokenizer, [], tokens)
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
        remove_columns = raw_dataset.column_names,
        fn_kwargs = {'tokenizer': tokenizer, 'max_length': chunk_size},
        **COMMON_DATASET_MAP_KWARGS,
    )
    train_dataset = code_dataset.map(
        prepare_documents_train_dataset,
        # remove_columns=code_dataset.column_names,
        fn_kwargs = {'tokenizer': tokenizer, 'train_methods': train_methods},
        **COMMON_DATASET_MAP_KWARGS,
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
        model: PreTrainedModel, tokenizer: PreTrainedTokenizer, config: DictConfig,
        train_dataset: Optional[Dataset] = None, eval_datasets: Optional[Dict[str, Dataset]] = None,
        compute_metrics = None, metrics_collator: Callable = None, **kwargs,
    ):
    """Train a model in a self-supervised manner given a dataset.

    This function serves as the backbone for the other training functions in this module.
    The dataset is expected to be fully formatted and ready for training.
    
    Args:
        model (PreTrainedModel): The model to train.
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
        config (dict): The training configuration.
        train_dataset (Dataset): The dataset to train on.
        eval_datasets (Dict[str, Dataset]): The datasets to evaluate on.
        compute_metrics (Callable): The function to compute metrics.
        metrics_collator (Callable): The function to collate metrics.
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

    metrics = None
    if train_dataset:
        trainer.train()
    elif eval_datasets:
        metrics = trainer.evaluate()
    else:
        raise ValueError("No dataset provided for training or evaluation!")

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
    
    return metrics

def train_self_supervised_documents(
        model: PreTrainedModel, tokenizer: PreTrainedTokenizer, config: DictConfig,
        documents: Optional[List[str]] = None, eval_data: Optional[List[str]] = None,
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
    if documents is None:
        train_dataset = None
    else:
        train_dataset = documents_to_dataset(documents, tokenizer, train_methods)

    if eval_data is None:
        eval_datasets = None
    else:
        eval_datasets = {}
        for dataset_name, data in eval_data.items():
            eval_datasets[dataset_name] = documents_to_dataset(data, tokenizer, train_methods)
        if len(eval_datasets) == 1:
            eval_datasets = eval_datasets[0]

    metrics = train_self_supervised(
        model, tokenizer, config, train_dataset, eval_datasets,
        compute_metrics, metrics_collator, **kwargs
    )

    return metrics


def train_self_supervised_project(
        model: PreTrainedModel, tokenizer: PreTrainedTokenizer, config: DictConfig,
        project_data = None, eval_data = None, compute_metrics = None,
        metrics_collator: Callable = None, train_methods: Optional[List[str]] = None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
    """Train a model in a self-supervised manner given a dataset.
    
    For arguments other than project_data, eval_data, and train_methods,
    see train_self_supervised.

    Args:
        project_data (Dict[str, str]): The project data to train on.
        eval_data (Dict[str, Dict[str, str]]): The evaluation data.
        train_methods (List[str]): The training methods to use.
            'ntp' (next token prediction) and 'fim' (fill-in-the-middle) are available.
            If None, only ntp is used.
    
    Returns:
        Optional[Dict[str, Any]]: The metrics from the evaluation (if no training set and evaluation only).
    """
    # Eval data must be in format {name_of_dataset: {file_name: file_contents, ...}},
    # even if only one eval dataset
    if project_data is None:
        train_dataset = None
    else:
        train_dataset = project_to_train_dataset(project_data, tokenizer, train_methods)

    if eval_data is None:
        eval_datasets = None
    else:
        eval_datasets = {}
        for dataset_name, data in eval_data.items():
            eval_datasets[dataset_name] = project_to_train_dataset(data, tokenizer, train_methods)
        if len(eval_datasets) == 1:
            eval_datasets = eval_datasets[list(eval_datasets.keys())[0]]

    metrics = train_self_supervised(
        model, tokenizer, config, train_dataset, eval_datasets,
        compute_metrics, metrics_collator, **kwargs
    )

    return metrics