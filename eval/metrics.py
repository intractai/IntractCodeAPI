from dataclasses import dataclass
import datetime
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig
import pandas as pd
import torch
from torch.nn import functional as F
from transformers.trainer_utils import EvalPrediction

sys.path.append('../')
from src.training.finetune import IGNORE_INDEX
from utils import *
from metrics import *


WANDB_PROJECT = 'backend_finetune_eval'


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
        reduction = 'none',
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


def compute_problem_level_metrics(
        eval_preds: EvalPrediction,
        fim_token_id: Optional[int] = None,
    ) -> Dict[str, float]:
    """Compute metrics for for each individual problem.
    
    Args:
        eval_preds: Evaluation predictions from a model.
        fim_token_id: Token ID for the FIM token. If None, FIM loss will not be calculated.
    
    Returns:
        Dictionary mapping metric names to metric values.
    """

    # logits: (batch_size, seq_len, vocab_size)
    # labels: (batch_size, seq_len)
    # inputs: (batch_size, seq_len)
    logits, labels, inputs = eval_preds

    # Shift logits and labels over by one
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    
    # Calculate the losses
    losses = F.cross_entropy(
        input = logits.swapaxes(1, 2),
        target = labels,
        ignore_index = IGNORE_INDEX,
        reduction = 'none',
    )

    # Get the average loss per sequence
    # losses: (batch_size, seq_len)
    valid_tokens_per_sequence = (labels != IGNORE_INDEX).sum(1)
    loss_per_sequence = losses.sum(1) / valid_tokens_per_sequence

    # Calculate accuracy
    preds = logits.argmax(2)
    correct_preds = (preds == labels)
    accuracy_per_sequence = correct_preds.sum(1) / valid_tokens_per_sequence

    # Get the problem type of each sequence (e.g. ntp or fim)
    if fim_token_id is not None:
        is_fim_sequence = (inputs == fim_token_id).any(1)
    else:
        is_fim_sequence = torch.zeros_like(valid_tokens_per_sequence, dtype=torch.bool)

    problem_types = ['fim' if is_fim else 'ntp' for is_fim in is_fim_sequence]

    return {
        'loss': loss_per_sequence.tolist(),
        'accuracy': accuracy_per_sequence.tolist(),
        'n_tokens': valid_tokens_per_sequence.tolist(),
        'problem_type': problem_types,
    }


def collate_problem_level_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Collate metrics from multiple runs in a way that preserves the metrics for each problem.

    Args:
        metrics_list: List of dictionaries mapping metric names to metric values.

    Returns:
        Dictionary mapping metric names to metric values.
    """
    collated_metrics = {k: [] for k in metrics_list[0].keys()}
    for metrics in metrics_list:
        for k, v in metrics.items():
            collated_metrics[k].extend(v)
    return collated_metrics


@dataclass
class EvalResults:
    """Results from running evaluation on a set of tasks and/or benchmarks."""
    task_metrics: Optional[List[Dict[str, Any]]] = None
    benchmark_metrics: Optional[List[Dict[str, Any]]] = None


def save_metrics(eval_results: EvalResults, config: DictConfig, logger: logging.Logger, output_dir: Optional[str] = None):
    """Save evaluation metrics to a file and/or wandb depending on the config.
    
    Args:
        eval_results: Results from running evaluation on a set of tasks and/or benchmarks.
        config: Configuration for saving metrics.
    """
    task_metrics_df = pd.DataFrame(eval_results.task_metrics) if eval_results.task_metrics else None
    benchmark_metrics_df = pd.DataFrame(eval_results.benchmark_metrics) if eval_results.benchmark_metrics else None

    output_dir = config['eval'].get('metrics_output_dir')
    if output_dir:
        logger.info(f"Saving metrics to {output_dir}...")
        # Get datetime for the output file
        if output_dir is None:
            now = datetime.datetime.now()
            output_dir = os.path.join(output_dir, now.strftime('%Y-%m-%d_%H-%M-%S'))

        os.makedirs(output_dir, exist_ok=True)

        # Save to csv
        if task_metrics_df is not None:
            task_metrics_df.to_csv(os.path.join(output_dir, 'task_metrics.csv'), index=False)

        if benchmark_metrics_df is not None:
            benchmark_metrics_df.to_csv(os.path.join(output_dir, 'benchmark_metrics.csv'), index=False)

    if config.get('wandb', False):
        logger.info("Logging to wandb...")
        import wandb
        wandb.init(project=WANDB_PROJECT, config=config)
        wandb.log({'task_metrics': task_metrics_df, 'benchmark_metrics': benchmark_metrics_df})