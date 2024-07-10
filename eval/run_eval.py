"""Evaluate the performance of finetuning on a set of sample projects and benchmarks.

Objectives of the eval:
  - Test how much finetuning on relevant materials improves performance on a project
  - Test how much finetuning on irrelevant projects hurts performance on a project
  - Test how much finetuning on arbitrary materials hurts performance on general coding benchmarks
"""

from dataclasses import dataclass
import datetime
import logging
import os
import random
import sys
from typing import Any, Dict, List

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from transformers import PreTrainedModel, PreTrainedTokenizer

sys.path.append('../')
from src import config_handler
from src.modeling import ModelProvider, FIM_HOLE_TOKEN
from src.routers.fine_tuner import finetune_model, ProjectFinetuneData
from src.training import finetune
from benchmarks import run_human_eval_benchmark
from data_loading import load_task_info
from utils import *
from metrics import *


VALID_BENCHMARKS = ['human_eval']


logger = logging.getLogger(__name__)


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


def run_eval(config: DictConfig, model_provider: ModelProvider):
    """Run evaluation steps for finetuning."""

    # Make a random seed that will be used for both pre- and post-finetune eval
    # This is required to keep the FIM examples the same for both runs
    eval_seed = random.randint(0, 2**32 - 1)

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


@hydra.main(version_base=None, config_path='../src/conf', config_name='eval')
def main(config: DictConfig):

    # Start logging and config
    configure_logging(logger)
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
    save_metrics(eval_results, config, logger)


if __name__ == '__main__':
    main()
