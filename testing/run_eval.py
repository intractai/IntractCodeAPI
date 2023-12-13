"""Test finetuning a model with the given arguments."""

import argparse
from argparse import Namespace
from pathlib import Path
import random
import sys
from typing import Dict, Optional

import numpy as np
import torch
from torch.nn import functional as F

sys.path.append('./')
from src import finetune, modeling
from src.finetune import IGNORE_INDEX
from src.arg_handling import parse_args


# Relative to this file
FINETUNE_DATA_DIR = 'data/finetune/'
OOD_DATA_DIR = 'data/ood/'


def parse_testing_args(raw_args: list[str]):
    """Parse arguments for testing finetuning."""

    parser = argparse.ArgumentParser()

    parser.add_argument('--holdout_frac', type=float, default=0.2,
                        help='Fraction of finetuning data per project ' + \
                             'to hold out for testing.')
    parser.add_argument('--total_epochs', type=int, default=2,
                        help='Total number of epochs over all data.')
    parser.add_argument('--epochs_per_project', type=int, default=1,
                        help='Number of epochs to train on each project.')
    parser.add_argument('--n_ood_projects', type=int, default=10,
                        help='Number of projects to use in OOD dataset.')
    parser.add_argument('--n_finetune_projects', type=int, default=10,
                        help='Number of projects to finetune on')
    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--wandb', action='store_true', default=False,
                        help='Whether to log to wandb.')

    parsed_args = parser.parse_args(raw_args)
    return parsed_args


def train(
        project_data: Dict[str, str],
        args: Namespace,
        finetune_args: Namespace):
    """Finetune a model on files from a given project.

    Args:
        project_data: Dictionary mapping file paths to file contents.
        args: Generic arguments for the model.
        finetune_args: Arguments for finetuning.
    """

    modeling.intialize_model(args.model_dir, args.local_model_dir, args)
    finetune.train_supervised_projectdir(
        project_data,
        output_dir=args.local_model_dir,
        report_to='none',
        **vars(finetune_args)
    )


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
                            project_data[file_path.name] = file_text
                    except UnicodeDecodeError:
                        print(f'Error reading file {file_path}. Skipping...')
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
    for project_name, project_data in multi_project_data.items():
        for file_name, file_data in project_data.items():
            all_data[f'{project_name}/{file_name}'] = file_data

    return all_data


def compute_metrics(eval_preds, fim_token_id):
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
    accuracy = (select_preds == select_labels).mean()

    # Calculate the mean loss
    losses = F.cross_entropy(
        input = torch.from_numpy(logits).swapaxes(1, 2),
        target = torch.from_numpy(labels),
        ignore_index = IGNORE_INDEX,
        reduction = 'none'
    )
    no_ignore_idxs = torch.from_numpy(no_ignore_idxs)
    # loss_sums = losses.sum(1)
    # n_targets = no_ignore_idxs.sum(1)
    # mean_losses = losses.sum(1) / no_ignore_idxs.sum(1)
    # loss = mean_losses.mean().item()

    # Check which inputs had the FIM token
    fim_idxs = (inputs == fim_token_id).any(1)
    # fim_loss = (losses[fim_idxs].sum(1) / n_targets[fim_idxs]).mean().item()
    fim_loss = (losses[fim_idxs].sum() / no_ignore_idxs[fim_idxs].sum()).item()
    ntp_loss = (losses[~fim_idxs].sum() / no_ignore_idxs[~fim_idxs].sum()).item()
    # combined_loss = (losses.sum() / no_ignore_idxs.sum()).item()

    loss_fn = torch.nn.CrossEntropyLoss()
    combined_loss = loss_fn(
        torch.from_numpy(logits).reshape(-1, logits.shape[-1]),
        torch.from_numpy(labels).reshape(-1)
    ).item()
    # combined_loss = F.cross_entropy(
    #     input = torch.from_numpy(logits).swapaxes(1, 2),
    #     target = torch.from_numpy(labels),
    #     ignore_index = IGNORE_INDEX,
    #     reduction = 'mean'
    # ).item()

    return {
        'accuracy': accuracy,
        'combined_loss': combined_loss,
        'fim_loss': fim_loss,
        'ntp_loss': ntp_loss
    }


def run_eval(
        args: Namespace,
        finetune_args: Namespace):
    """Run evaluation steps for finetuning."""

    # Load data
    print('Loading project data...')
    primary_data = load_projects(FINETUNE_DATA_DIR, args.n_finetune_projects)
    ood_data = load_projects(OOD_DATA_DIR, args.n_ood_projects)

    # Flatten the ood_data
    all_ood_data = combine_project_data(ood_data)

    # # Split primary data into training and testing sets
    # primary_train_data = {}
    # primary_test_data = {}
    # for project_name, project_data in primary_data.items():
    #     train_data, test_data = split_project_data(project_data, args.holdout_frac)
    #     primary_train_data[project_name] = train_data
    #     primary_test_data[project_name] = test_data

    # Determine project training order for `finetune_data`
    finetune_order = list(primary_data.keys())
    np.random.shuffle(finetune_order)

    # Preare the compute metrics function
    tokenizer = modeling.GLOBAL_TOKENIZER
    fim_token_id = tokenizer.vocab[finetune.FIM_HOLE_TOKEN]
    metrics_fn = lambda eval_preds: compute_metrics(eval_preds, fim_token_id)

    # Train for an epoch on each primary project while periodically logging metrics
    print('Starting training...')
    for outer_epoch in range(args.total_epochs):
        for project_idx, project_name in enumerate(finetune_order):

            # Get the data for the current project we are finetuning on
            current_project_data = primary_data[project_name]

            # Get the data for all projects we have seen previously
            if outer_epoch == 0:
                seen_project_data = {
                    project_name: primary_data[project_name]
                    for project_name in finetune_order[:project_idx]
                }
                seen_project_data = combine_project_data(seen_project_data)
            else:
                seen_project_data = combine_project_data(primary_data)

            # Gather the eval datasets
            eval_datasets = {'ood': all_ood_data}
            if len(seen_project_data) > 0:
                eval_datasets['previous_projects'] = seen_project_data

            # print('ood', len(all_ood_data), [len(v) for k, v in all_ood_data.items()])
            # print('=' * 100)
            # print('primary', len(current_project_data), [len(v) for k, v in current_project_data.items()])
            # print('=' * 100)
            # print('previous_projects', len(seen_project_data), [len(v) for k, v in seen_project_data.items()])
            # exit()

            train_args = dict(
                project_data = current_project_data,
                eval_data = eval_datasets,
                output_dir = args.local_model_dir,
                num_train_epochs = args.epochs_per_project,
                report_to = 'wandb' if args.wandb else 'none',
                do_eval = True,
                evaluation_strategy = 'epoch',
                compute_metrics = metrics_fn
            )
            train_args = {**train_args, **vars(finetune_args)}

            # Finetune on the current project
            print(f'Training on project {project_name}...')
            finetune.train_supervised_projectdir(**train_args)


if __name__ == '__main__':
    generic_args, env_args, unknown_args = parse_args(env_prefixes=['FINETUNE_'])
    testing_args = parse_testing_args(unknown_args)
    args = Namespace(**{**vars(generic_args), **vars(testing_args)})

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load model
    modeling.initialize_model(args.model_dir, args.local_model_dir, args)

    # Finetune and evaluate
    run_eval(args, env_args['finetune'])
