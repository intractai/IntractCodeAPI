"""Evaluate the performance of finetuning on a set of sample projects and benchmarks.

Objectives of the eval:
  - Test how much finetuning on relevant materials improves performance on a project
  - Test how much finetuning on irrelevant projects hurts performance on a project
  - Test how much finetuning on arbitrary materials hurts performance on general coding benchmarks
"""

from dataclasses import dataclass
import datetime
import json
import logging
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import warnings

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
from torch.nn import functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.trainer_utils import EvalPrediction

sys.path.append('../')
from src import config_handler
# from src import finetune, modeling
# from src.data_formatting import IGNORE_INDEX, FIM_HOLE_TOKEN
from src.modeling import ModelProvider, FIM_HOLE_TOKEN
from src.routers.fine_tuner import finetune_model, ProjectFinetuneData
from src.training import finetune
from src.training.finetune import IGNORE_INDEX
from benchmarks import run_human_eval_benchmark


# Relative to this file
DATA_DIR = 'data/'
METADATA_FILENAME = 'metadata.json'
TASK_TRAIN_DIR = 'train/'
TASK_TEST_DIR = 'test/'
TASK_CODE_DIR = 'code/'
TASK_DOCS_DIR = 'documents/'
TASK_LINKS_FILENAME = 'links.txt'

VALID_BENCHMARKS = ['human_eval']

WANDB_PROJECT = 'backend_finetune_eval'


logger = logging.getLogger(__name__)


def combine_project_data(multi_project_data: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """Combine data from multiple projects into a single dictionary.

    Args:
        multi_project_data: Dictionary mapping project names to dictionaries mapping
            file names to file contents.

    Returns:
        Dictionary mapping file names to file contents.
    """

    all_data = {}
    for project_name, project_data in multi_project_data.items():
        for file_path, file_data in project_data.items():
            all_data[os.path.join(project_name, file_path)] = file_data

    return all_data


def load_projects(
        relative_path: str, combine_projects: bool = True,
    ) -> Union[Dict[str, Dict[str, str]], Dict[str, str]]:
    """Load project data from a directory.

    Args:
        relative_path: Path to directory containing projects.
        combine_projects: Whether to combine data from multiple projects into a single, flat dictionary.

    Returns:
        Dictionary mapping project names to dictionaries mapping file names to file contents.
        If `combine_projects` is True, returns a single dictionary mapping file names to file contents.
    """
    all_project_data = {}
    data_dir = Path(__file__).parent / relative_path
    # Loop through each project in this directory
    for i, project_path in enumerate(data_dir.glob('*')):
        if project_path.is_dir():
            project_data = {}
            # Loop through each file in this project
            for file_path in project_path.rglob('*'):
                if file_path.is_file():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_text = f.read()
                        if len(file_text) > 0:
                            # Path to file starting from the name of the project folder
                            rel_path = file_path.relative_to(project_path.parent)
                            project_data[str(rel_path)] = file_text
                    except UnicodeDecodeError:
                        logging.warning(f'Error reading file {file_path}. Skipping...')
            all_project_data[project_path.name] = project_data

    if combine_projects:
        all_project_data = combine_project_data(all_project_data)

    return all_project_data


def load_documents(relative_path: str) -> List[str]:
    """Load documents from a directory.

    Args:
        relative_path: Path to directory containing documents.

    Returns:
        List of document texts.
    """
    all_documents = []
    data_dir = Path(__file__).parent / relative_path
    for doc_path in data_dir.glob('**/*'):
        if doc_path.is_file():
            with open(doc_path, 'rb') as f:
                doc_bytes = f.read()
            if len(doc_bytes) > 0:
                all_documents.append((doc_path.name, doc_bytes))
                logger.debug(f'Loaded document: {doc_path}')

    return all_documents


def load_links(relative_path: str) -> List[str]:
    """Load links from a file.

    Args:
        relative_path: Path to file containing links.

    Returns:
        List of links.
    """
    all_links = []
    data_dir = Path(__file__).parent / relative_path
    with open(data_dir, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) > 0:
                all_links.append(line.strip())

    return all_links


def load_task_info() -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Load all the info (code, documents, and links) for a single eval task.
    Args:
        relative_path: Path to file containing task information.

    Returns:
        Tuple of task name and dictionary containing task information.
    """
    data_dir = Path(__file__).parent / DATA_DIR
    # Get all folders in the directory (1 folder = 1 task)
    for task_path in data_dir.glob('*'):
        if task_path.is_dir():
            task_info = {'train': {}, 'test': {}, 'metadata': {}}
            task_name = task_path.name

            # Paths to relevant files / directories
            metadata_path = task_path / METADATA_FILENAME
            train_path = task_path / TASK_TRAIN_DIR
            code_path = train_path / TASK_CODE_DIR
            docs_path = train_path / TASK_DOCS_DIR
            links_path = train_path / TASK_LINKS_FILENAME
            test_path = task_path / TASK_TEST_DIR

            # Load metadata
            if metadata_path.is_file():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                task_info['metadata'] = metadata

            # Load train data
            if code_path.is_dir():
                task_info['train']['code'] = load_projects(code_path)
            if docs_path.is_dir():
                task_info['train']['docs'] = load_documents(docs_path)
            if links_path.is_file():
                task_info['train']['links'] = load_links(links_path)

            # Load test data
            if test_path.is_dir():
                task_info['test'] = load_projects(test_path)

            if len(task_info['train']) == 0:
                logger.warning(f'Task {task_name} has no train data. Skipping...')
                continue
            elif len(task_info['test']) == 0:
                logger.warning(f'Task {task_name} has no test data. Skipping...')
                continue

            yield task_name, task_info


def compute_metrics(
        eval_preds: EvalPrediction,
        fim_token_id: Optional[int] = None,
    ) -> Dict[str, float]:
    """Compute metrics for finetuning.
    
    Args:
        eval_preds: Evaluation predictions from a model.
        fim_token_id: Token ID for the FIM token. If None, FIM loss will not be calculated.
    
    Returns:
        Dictionary mapping metric names to metric values.
    """

    logits, labels, inputs = eval_preds

    # Shift logits and labels over by one
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]

    # Gather all preds and labels where the label is not -100
    preds = logits.argmax(2)
    no_ignore_idxs = (labels != IGNORE_INDEX)

    select_preds = preds[no_ignore_idxs]
    select_labels = labels[no_ignore_idxs]

    # Calculate the accuracy
    n_correct = (select_preds == select_labels).sum()
    n_tokens = no_ignore_idxs.sum()

    # Calculate the mean loss
    losses = F.cross_entropy(
        input = logits.swapaxes(1, 2),
        target = labels,
        ignore_index = IGNORE_INDEX,
        reduction = 'none'
    )

    if fim_token_id is not None:
        # Check which inputs had the FIM token
        fim_idxs = (inputs == fim_token_id).any(1)
        # Calculate how many tokens there are in each category
        n_fim_tokens = no_ignore_idxs[fim_idxs].sum()
        # Calculate the total loss for each category
        fim_loss = losses[fim_idxs].sum()
    else:
        n_fim_tokens = torch.tensor(float('nan'))
        fim_loss = torch.tensor(float('nan'))

    n_ntp_tokens = no_ignore_idxs[~fim_idxs].sum()
    ntp_loss = losses[~fim_idxs].sum()

    return {
        'n_tokens': n_tokens,
        'n_fim_tokens': n_fim_tokens,
        'n_ntp_tokens': n_ntp_tokens,
        'n_correct': n_correct,
        'fim_loss': fim_loss,
        'ntp_loss': ntp_loss,
        # 'accuracy': accuracy,
        # 'combined_loss': combined_loss,
    }


def collate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Collate metrics from multiple runs.

    Args:
        metrics_list: List of dictionaries mapping metric names to metric values.

    Returns:
        Dictionary mapping metric names to metric values.
    """

    # Initialize the collated metrics
    collated_metrics = {}
    sum_metrics = {}

    # First, sum each of the metrics
    for metric_name in metrics_list[0].keys():
        sum_metrics[metric_name] = 0
        for metrics in metrics_list:
            sum_metrics[metric_name] += metrics[metric_name]

    collated_metrics['accuracy'] = sum_metrics['n_correct'] / sum_metrics['n_tokens']
    collated_metrics['fim_loss'] = sum_metrics['fim_loss'] / sum_metrics['n_fim_tokens']
    collated_metrics['ntp_loss'] = sum_metrics['ntp_loss'] / sum_metrics['n_ntp_tokens']
    collated_metrics['combined_loss'] = \
        (sum_metrics['fim_loss'] + sum_metrics['ntp_loss']) / sum_metrics['n_tokens']

    return collated_metrics


def create_new_model_tuple(model_provider: ModelProvider):
    """Create a new model and tokenizer."""
    model, model_utils = model_provider.create_new_model_tuple()
    tokenizer = model_utils['tokenizer']
    return model, tokenizer


def run_benchmarks(
        config: DictConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        names: List[str] = None,
    ) -> Dict[str, Any]:
    """Run all benchmarks specified in the config.
    
    When names is None, or contains 'all', run all benchmarks specified in the config.
    Otherwise, run only the benchmarks specified in names.
    """
    if names is not None and 'all' in names:
        names = None

    metrics = {}
    enabled_benchmarks = set([b for b in VALID_BENCHMARKS if config['eval'].get(b, False)])
    for benchmark_name in enabled_benchmarks:
        if names is not None and benchmark_name not in names:
            continue

        if benchmark_name == 'human_eval':
            logger.info('Running human eval benchmark...')
            metrics['human_eval_score'] = run_human_eval_benchmark(config, model, tokenizer)
        else:
            raise ValueError(f'Unknown benchmark: {benchmark_name}')
        
    return metrics


def run_task_eval(
        config: DictConfig,
        task_info: Dict[str, Any],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ) -> Dict[str, Any]:
    """Run evaluation on a single task and return a dictionary of metrics."""
    train_methods = []
    if config.train.use_ntp:
        train_methods.append('ntp')
    if config.train.use_fim:
        train_methods.append('fim')

    fim_token_id = tokenizer.vocab[FIM_HOLE_TOKEN]
    compute_metrics_fn = lambda eval_preds: compute_metrics(eval_preds, fim_token_id)

    metrics = finetune.train_self_supervised_project(
        model,
        tokenizer,
        config.train,
        project_data = None,
        eval_data = {'eval_code': task_info['test']},
        compute_metrics = compute_metrics_fn,
        metrics_collator = collate_metrics,
        output_dir = config.model.save_model_dir,
        report_to = 'none',
        train_methods = train_methods,
    )

    return metrics


@dataclass
class EvalResults:
    """Results from running evaluation on a set of tasks and/or benchmarks."""
    task_metrics: List[Dict[str, Any]]
    benchmark_metrics: List[Dict[str, Any]]


def run_eval(config: DictConfig, model_provider: ModelProvider):
    """Run evaluation steps for finetuning."""

    ### Run benchmarks ###

    benchmark_metrics = []
    enabled_benchmarks = set([b for b in VALID_BENCHMARKS if config['eval'].get(b, False)])
    if len(enabled_benchmarks) > 0:
        logger.info(f"Enabled benchmarks: {enabled_benchmarks}")
        model, tokenizer = create_new_model_tuple(model_provider)
        new_metrics = run_benchmarks(config, model, tokenizer)
        logger.debug(f"Pre-finetune benchmark metrics: {new_metrics}")
        for benchmark_name, score in new_metrics.items():
            benchmark_metrics.append({
                'benchmark': benchmark_name,
                'score': score,
                'finetuned': False,
                'finetune_task': None,
            })
        del model, tokenizer
    else:
        logger.info("No benchmarks enabled.")

    if not config['eval'].get('custom_tasks', True):
        return EvalResults([], benchmark_metrics)

    # Make a random seed that will be used for both pre- and post-finetune eval
    # This is required to keep the FIM examples the same for both runs
    eval_seed = random.randint(0, 2**32 - 1)

    # Load name of all eval tasks
    task_metrics = []
    for task_name, task_info in load_task_info():
        logger.info(f"===== Starting eval on task [{task_name}] =====")

        # Reinit a new model
        model, tokenizer = create_new_model_tuple(model_provider)


        ### Run eval ###

        logger.info(f"Running pre-finetune eval...")
        set_seed(eval_seed)
        eval_metrics = run_task_eval(config, task_info, model, tokenizer)
        eval_metrics['task_name'] = task_name
        eval_metrics['finetuned'] = False
        eval_metrics.update(task_info['metadata'])
        task_metrics.append(eval_metrics)
        logger.debug(f"Pre-finetune eval metrics for task {task_name}: {eval_metrics}")


        ### Finetune ###

        logger.info(f"Finetuning...")
        # TODO: Document data is currently unused, add then to finetuning eventually
        finetune_data = ProjectFinetuneData(
            project_dict = task_info['train'].get('code'),
            urls = task_info['train'].get('links'),
            language = 'python',
            documents = task_info['train'].get('docs'),
        )
        model = finetune_model(finetune_data, config, model, tokenizer)


        ### Run eval again ###

        logger.info(f"Running post-finetune eval...")
        set_seed(eval_seed)
        eval_metrics = run_task_eval(config, task_info, model, tokenizer)
        eval_metrics['task_name'] = task_name
        eval_metrics['finetuned'] = True
        eval_metrics.update(task_info['metadata'])
        task_metrics.append(eval_metrics)
        logger.debug(f"Post-finetune eval metrics for task {task_name}: {eval_metrics}")


        ### Run benchmarks if specified ###

        if task_info['metadata'].get('benchmarks', '').lower() == 'all':
            target_benchmarks = enabled_benchmarks
        else:
            target_benchmarks = enabled_benchmarks.intersection(
                set(task_info['metadata'].get('benchmarks', []))
            )
        if len(target_benchmarks) > 0:
            logger.info(f"Running benchmarks...")
            new_metrics = run_benchmarks(
                config, model, tokenizer, names=target_benchmarks)
            logger.debug(f"Post-finetune benchmark metrics for task {task_name}: {new_metrics}")
            for benchmark_name, score in new_metrics.items():
                benchmark_metrics.append({
                    'benchmark': benchmark_name,
                    'score': score,
                    'finetuned': True,
                    'finetune_task': task_name,
                })
        else:
            logger.info("No benchmarks to run.")
        
        # Delete the model and tokenizer for the task
        del model, tokenizer

    return EvalResults(task_metrics, benchmark_metrics)


def save_metrics(eval_results: EvalResults, config: DictConfig):
    """Save evaluation metrics to a file and/or wandb depending on the config.
    
    Args:
        eval_results: Results from running evaluation on a set of tasks and/or benchmarks.
        config: Configuration for saving metrics.
    """
    task_metrics_df = pd.DataFrame(eval_results.task_metrics)
    benchmark_metrics_df = pd.DataFrame(eval_results.benchmark_metrics)

    output_dir = config['eval'].get('metrics_output_dir')
    if output_dir:
        logger.info(f"Saving metrics to {output_dir}...")
        # Get datetime for the output file
        now = datetime.datetime.now()
        output_dir = os.path.join(output_dir, now.strftime('%Y-%m-%d_%H-%M-%S'))

        os.makedirs(output_dir, exist_ok=True)

        # Save to csv
        task_metrics_df.to_csv(os.path.join(output_dir, 'task_metrics.csv'), index=False)
        benchmark_metrics_df.to_csv(os.path.join(output_dir, 'benchmark_metrics.csv'), index=False)

    if config.get('wandb', False):
        logger.info("Logging to wandb...")
        import wandb
        wandb.init(project=WANDB_PROJECT, config=config)
        wandb.log({'task_metrics': task_metrics_df, 'benchmark_metrics': benchmark_metrics_df})


def configure_logging():
    """Configure logging for the server."""
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # add log in the file
    handler = logging.FileHandler('../eval_log.txt')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    warnings.filterwarnings('ignore', message=".*Could not load referrer policy.*")
    trafilatura_logger = logging.getLogger('trafilatura')
    trafilatura_logger.setLevel(logging.INFO)
    lite_llm = logging.getLogger('LiteLLM')
    lite_llm.setLevel(logging.INFO)


def set_seed(seed: Optional[int] = None):
    """Set the random seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


@hydra.main(version_base=None, config_path='../src/conf', config_name='eval')
def main(config: DictConfig):

    # Start logging and config
    configure_logging()
    config = OmegaConf.create(config)
    logger.info(f"Loaded config: {config}")

    # Initialize singletons
    config_handler.ConfigProvider.initialize(config)
    model_provider = ModelProvider.get_instance(config.model)

    # Set random seed
    set_seed(config.get('seed'))
    
    # Run eval
    logger.info("Running eval...")
    eval_results = run_eval(config, model_provider)
    save_metrics(eval_results, config)


if __name__ == '__main__':
    main()
