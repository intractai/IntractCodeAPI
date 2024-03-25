"""Test finetuning a model with the given arguments."""

from argparse import Namespace
import logging
from pathlib import Path
import random
import sys
from typing import Dict, List, Optional

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch.nn import functional as F
from transformers.trainer_utils import EvalPrediction

sys.path.append('./')
from src import config, finetune, modeling
from src.data_formatting import IGNORE_INDEX, FIM_HOLE_TOKEN


# Relative to this file
FINETUNE_DATA_DIR = 'data/finetune/'
OOD_DATA_DIR = 'data/ood/'


logger = logging.getLogger(__name__)


# Run eval steps:
#   1. Decide on a project training order
#   2. Train for an epoch on each primary project while periodically logging metircs
# Metrics to log:
#   - Combined loss
#   - FIM loss
#   - Next token prediction loss
# Metric categories:
#   - Performance on OOD data
#   - Performance on seen primary project data
#   - Performance on held out primary project data
#   - Performance on previously seen primary project data
# Tunable hyperparameters:
#   - Number of epochs per project
#   - Percent FIM
#   - Whether to prefix prompts with file path & line number


def load_projects(relative_path: str, n: Optional[int] = None) -> Dict[str, Dict[str, str]]:
    """Load project data from a directory.

    Args:
        relative_path: Path to directory containing projects.
        n: Number of projects to load. If None, load all projects.

    Returns:
        Dictionary mapping project names to dictionaries mapping file names to file contents.
    """

    # Load project data
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

        if n is not None and i >= n:
            break

    return all_project_data


def split_project_data(project_data: Dict[str, str], holdout_frac: float):
    """Split project data into training and testing sets.

    Args:
        project_data: Dictionary mapping file names to file contents.
        holdout_frac: Fraction of finetuning data per project to hold out for testing.

    Returns:
        Tuple of dictionaries mapping project names to dictionaries mapping file names
        to file contents. The first dictionary contains the training data, and the
        second contains the testing data.
    """

    # Determine which files to hold out
    all_files = list(project_data.keys())
    num_holdout_files = int(len(all_files) * holdout_frac)
    holdout_files = set(np.random.choice(all_files, num_holdout_files, replace=False))

    # Split data into training and testing sets
    train_data = {}
    test_data = {}
    for file_name, file_data in project_data.items():
        if file_name in holdout_files:
            test_data[file_name] = file_data
        else:
            train_data[file_name] = file_data

    return train_data, test_data


def combine_project_data(multi_project_data: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """Combine data from multiple projects into a single dictionary.

    Args:
        multi_project_data: Dictionary mapping project names to dictionaries mapping
            file names to file contents.

    Returns:
        Dictionary mapping file names to file contents.
    """

    all_data = {}
    for project_data in multi_project_data.values():
        for file_path, file_data in project_data.items():
            all_data[file_path] = file_data

    return all_data


def compute_metrics(
        eval_preds: EvalPrediction,
        fim_token_id: int
    ) -> Dict[str, float]:
    """Compute metrics for finetuning.
    
    Args:
        eval_preds: Evaluation predictions from a model.
        fim_token_id: ID of the FIM hole token.
    
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

    # Check which inputs had the FIM token
    fim_idxs = (inputs == fim_token_id).any(1)

    # Calculate how many tokens there are in each category
    n_fim_tokens = no_ignore_idxs[fim_idxs].sum()
    n_ntp_tokens = no_ignore_idxs[~fim_idxs].sum()

    # Calculate the total loss for each category
    fim_loss = losses[fim_idxs].sum()
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
        # 'fim_loss': fim_loss,
        # 'ntp_loss': ntp_loss
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


def run_eval(cfg: Namespace):
    """Run evaluation steps for finetuning."""

    eval_cfg = cfg.eval_cfg

    # Load data
    logging.info('Loading project data...')
    primary_data = load_projects(FINETUNE_DATA_DIR, eval_cfg.n_finetune_projects)
    ood_data = load_projects(OOD_DATA_DIR, eval_cfg.n_ood_projects)

    # Flatten the ood_data
    all_ood_data = combine_project_data(ood_data)

    # Split primary data into training and testing sets
    primary_train_data = {}
    primary_test_data = {}
    for project_name, project_data in primary_data.items():
        train_data, test_data = split_project_data(project_data, eval_cfg.holdout_frac)
        if len(train_data) == 0 or len(test_data) == 0:
            logging.warning(f'Project {project_name} has no data. Skipping...')
            continue
        primary_train_data[project_name] = train_data
        primary_test_data[project_name] = test_data

    # Determine project training order for `finetune_data`
    finetune_order = list(primary_train_data.keys())
    np.random.shuffle(finetune_order)

    # Preare the compute metrics function
    mp = modeling.ModelProvider.get_instance()
    tokenizer = mp.get_model_utils()['tokenizer']
    fim_token_id = tokenizer.vocab[FIM_HOLE_TOKEN]

    metrics_fn = lambda eval_preds: compute_metrics(eval_preds, fim_token_id)

    # Train for an epoch on each primary project while periodically logging metrics
    logging.info('Starting training...')
    for outer_epoch in range(eval_cfg.total_epochs):
        for project_idx, project_name in enumerate(finetune_order):

            # Get the data for the current project we are finetuning on
            curr_train_set = primary_train_data[project_name]
            curr_test_set = primary_test_data[project_name]

            # Get the data for all projects we have seen previously
            if outer_epoch == 0:
                seen_train_set = {
                    project_name: primary_train_data[project_name]
                    for project_name in finetune_order[:project_idx]
                }
                seen_test_set = {
                    project_name: primary_train_data[project_name]
                    for project_name in finetune_order[:project_idx]
                }
                seen_train_set = combine_project_data(seen_train_set)
                seen_test_set = combine_project_data(seen_test_set)
            else:
                seen_train_set = combine_project_data(primary_train_data)
                seen_test_set = combine_project_data(primary_test_data)

            # Gather the eval datasets
            eval_datasets = {
                'ood': all_ood_data,
                'curr_project_holdout': curr_test_set,
            }
            if len(seen_train_set) > 0:
                eval_datasets['previous_projects_train'] = seen_train_set
                eval_datasets['previous_projects_holdout'] = seen_test_set

            train_args = dict(
                do_eval = True,
                logging_first_step = True,
                evaluation_strategy = 'epoch',
                output_dir = cfg.model_cfg.save_model_dir,
                report_to = 'wandb' if cfg.get('wandb') else 'none'
            )

            # Update args for HF Trainer
            finetune_cfg = cfg.train_cfg.copy()
            OmegaConf.set_struct(finetune_cfg, False)
            finetune_cfg.update(train_args)
            OmegaConf.set_struct(finetune_cfg, True)

            # Finetune on the current project
            logging.info(f'Training on project {project_name}...')
            finetune.train_supervised_projectdir(
                project_data = curr_train_set,
                eval_data = eval_datasets,
                compute_metrics = metrics_fn,
                metrics_collator = collate_metrics,
                train_cfg = finetune_cfg
            )


def configure_logging():
    """Configure logging for the server."""
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # add log in the file
    handler = logging.FileHandler("log.txt")
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@hydra.main(version_base=None, config_path="../app/conf", config_name="eval")
def main(cfg: DictConfig):
    # Initialize logging and get config
    configure_logging()
    cfg = OmegaConf.create(cfg)
    modeling.ModelProvider(cfg.model_cfg)
    config.ConfigProvider.initialize(cfg)

    # Set random seed
    if cfg.get('seed') is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    # Run eval
    run_eval(cfg)


if __name__ == '__main__':
    main()
