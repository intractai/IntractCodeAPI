"""Test finetuning a model with the given arguments."""

from argparse import Namespace
import os
from pathlib import Path
import random
import sys
from typing import Dict

sys.path.append('./')
from src import finetune, modeling
from src.arg_handling import parse_args


# Relative to this file
FINETUNE_DATA_DIR = 'data/finetune/'
OOD_DATA_DIR = 'data/ood/'


# def parse_testing_args():
#     # n finetun


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

def load_projects(relative_path: str) -> Dict[str, Dict[str, str]]:
    """Load project data from a directory.

    Args:
        relative_path: Path to directory containing projects.

    Returns:
        Dictionary mapping project names to dictionaries mapping file names to file contents.
    """

    # Load project data
    all_project_data = {}
    data_dir = Path(__file__).parent / relative_path
    # Loop through each project in this directory
    for project_path in data_dir.glob('*'):
        if project_path.is_dir():
            project_data = {}
            # Loop through each file in this project
            for file_path in project_path.rglob('*'):
                if file_path.is_file():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            project_data[file_path.name] = f.read()
                    except UnicodeDecodeError:
                        print(f'Error reading file {file_path}. Skipping...')
            all_project_data[project_path.name] = project_data

    return all_project_data


def run_eval(
        args: Namespace,
        finetune_args: Namespace):
    """Run evaluation steps for finetuning."""

    # Load data
    print('Loading project data...')
    finetune_data = load_projects(FINETUNE_DATA_DIR)
    ood_data = load_projects(OOD_DATA_DIR)

    # Determine project training order for `finetune_data`
    finetune_order = list(finetune_data.keys())
    random.shuffle(finetune_order)

    # Train for an epoch on each primary project while periodically logging metrics
    print('Starting training...')
    for project_name in finetune_order:
        current_project_data = finetune_data[project_name]
        print(f'Training on project {project_name}...')
        finetune.train_supervised_projectdir(
            project_data=current_project_data,
            eval_dataset=ood_data[list(ood_data.keys())[0]],
            output_dir=args.local_model_dir,
            report_to='none',
            **vars(finetune_args)
        )


if __name__ == '__main__':
    args, env_args, unknown_args = parse_args(env_prefixes=['FINETUNE_'])
    modeling.intialize_model(args.model_dir, args.local_model_dir, args)
    run_eval(args, env_args['finetune'])
