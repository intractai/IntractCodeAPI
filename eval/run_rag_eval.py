"""Evaluate the performance of RAG.

Objectives of the eval:
  - Measure the overall impact of RAG on generation performance.
  - Measure how well the model is able to utilize information from difference types of sources.
  - Measure the gap in current performance vs. potential performance that stems from:
    - Defficiencies in retrieval.
    - Defficiencies in how the information is stored (e.g. chunking).
"""

from dataclasses import dataclass
import datetime
import logging
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple
import warnings

import hydra
from llama_index.core import VectorStoreIndex
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
from src.rag import retrieve_context, VectorStoreProvider
from src.routers.fine_tuner import collect_item_data, finetune_model, ProjectFinetuneData
from src.training import finetune
from src.training.data_formatting import format_rag_query
from src.training.finetune import (
    COMMON_DATASET_MAP_KWARGS,
    problems_to_train_samples,
    project_to_problems,
    train_self_supervised,
) 
from data_loading import load_task_info
from utils import *
from metrics import (
    collate_problem_level_metrics,
    compute_problem_level_metrics,
    EvalResults,
    save_metrics,
)


logger = logging.getLogger(__name__)


def prepare_vector_store(
        config: DictConfig,
        vs_provider: VectorStoreProvider,
        task_info: Dict[str, Any]
    ) -> Tuple[VectorStoreIndex, int]:
    """Create a vector store and populate it with documents from the task.
    
    Args:
        config (DictConfig): Top level config.
        vs_provider (VectorStoreProvider): Vector store provider used to create a new vector store.
        task_info (Dict[str, Any]): Task info dictionary with 'code', 'docs', and 'links' keys.
    
    Returns:
        Tuple[VectorStoreIndex, int]: The populated vector store and the number of documents added.
    """
    vector_store = vs_provider.create_new_vector_store()

    finetune_data = ProjectFinetuneData(
        project_dict = task_info.get('code'),
        documents = task_info.get('docs'),
        urls = task_info.get('links'),
    )
    documents = collect_item_data(finetune_data, config)
    if len(documents) > 0:
        vs_provider._add_documents_to_vs(vector_store, documents)

    return vector_store, len(documents)


def calculate_weighted_metrics(prefix: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate the average weighted loss and accuracy for a set of sample-level statistics.

    Args:
        prefix (str): Prefix for the metrics to calculate.
        metrics (Dict[str, Any]): Dictionary of metrics to calculate the weighted averages for.

    Returns:
        Dict[str, Any]: Dictionary of the calculated metrics.
    """
    output_metrics = {}

    n_tokens = torch.tensor(metrics[f'{prefix}/n_tokens'])
    problem_type = metrics[f'{prefix}/problem_type']
    losses = torch.tensor(metrics[f'{prefix}/loss'])
    accuracies = torch.tensor(metrics[f'{prefix}/accuracy'])

    ntp_mask = torch.tensor([x == 'ntp' for x in problem_type])
    total_ntp_tokens = n_tokens[ntp_mask].sum()
    ntp_weights = n_tokens[ntp_mask] / total_ntp_tokens

    fim_mask = torch.tensor([x == 'fim' for x in problem_type])
    total_fim_tokens = n_tokens[fim_mask].sum()
    fim_weights = n_tokens[fim_mask] / total_fim_tokens

    total_tokens = n_tokens.sum()
    weights = n_tokens / total_tokens

    output_metrics[f'ntp_loss'] = torch.sum(
        losses[ntp_mask] * ntp_weights)
    output_metrics[f'fim_loss'] = torch.sum(
        losses[fim_mask] * fim_weights)
    output_metrics[f'loss'] = torch.sum(
        losses * weights)
    
    output_metrics[f'ntp_accuracy'] = torch.sum(
        accuracies[ntp_mask] * ntp_weights)
    output_metrics[f'fim_accuracy'] = torch.sum(
        accuracies[fim_mask] * fim_weights)
    output_metrics[f'accuracy'] = torch.sum(
        accuracies * weights)
    
    return {k: v.tolist() for k, v in output_metrics.items()}


def run_task_eval(
        config: DictConfig,
        task_info: Dict[str, Any],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        vector_store: VectorStoreIndex,
    ) -> Dict[str, Any]:
    """Run evaluation on a single task and return a dictionary of metrics."""
    
    train_methods = []
    if config.train.use_ntp:
        train_methods.append('ntp')
    if config.train.use_fim:
        train_methods.append('fim')

    # Convert task data into individual problems (leave space for context)
    problems_dataset = project_to_problems(
        project_data = task_info['test'],
        tokenizer = tokenizer,
        train_methods = train_methods,
        context_size = config.model.context_length - config.rag.max_gen_context_length,
    )

    # Query top k relevant contexts for each problem
    retriever = vector_store.as_retriever(similarity_top_k=config.eval.top_k_retrievals)
    
    problem_contexts = [] # (n_problems, n_retrievals)
    for row in problems_dataset:
        rag_query = format_rag_query(
            row['prior_context'], row['proceeding_context'], config.rag.max_embed_context_length)
        retrievals = retrieve_context(rag_query, vector_store, retriever=retriever)
        retrievals = retrievals or ['' for _ in range(config.eval.top_k_retrievals)]
        problem_contexts.append(retrievals)

    # For each problem, create a new k problems, each with a different retrieved context
    # Each dataset contains the same problem with a different set of retrievals
    context_template = config.rag.context_for_generation_template

    rag_problem_datasets = [problems_dataset]
    for i in range(config.eval.top_k_retrievals):
        retrievals = [context_template.format(contexts[i]) for contexts in problem_contexts]
        new_dataset = problems_dataset.add_column('additional_context', retrievals)
        rag_problem_datasets.append(new_dataset)

    # Tokenize the datasets and convert into training format
    eval_datasets = {}
    for i, dataset in enumerate(rag_problem_datasets):
        train_dataset = dataset.map(
            problems_to_train_samples,
            fn_kwargs = {
                'tokenizer': tokenizer,
                'additional_context_max_length': config.rag.max_gen_context_length,
            },
            remove_columns = dataset.column_names,
            **COMMON_DATASET_MAP_KWARGS,
        )
        eval_datasets[i] = train_dataset

    # Run evaluation on all of the problems, and measure the metrics for each

    fim_token_id = tokenizer.vocab[FIM_HOLE_TOKEN]
    compute_metrics_fn = lambda eval_preds: compute_problem_level_metrics(eval_preds, fim_token_id)

    metrics = train_self_supervised(
        model,
        tokenizer,
        config.train,
        train_dataset = None,
        eval_datasets = eval_datasets,
        compute_metrics = compute_metrics_fn,
        metrics_collator = collate_problem_level_metrics,
        report_to = 'none',
        output_dir = config.model.save_model_dir,
    )

    # Calculate the best metrics from all different contexts for each problem

    all_losses = torch.stack([
        torch.tensor(metrics[f'eval_{i}/loss'])
        for i in range(config.eval.top_k_retrievals + 1)
    ])
    all_accuracies = torch.stack([
        torch.tensor(metrics[f'eval_{i}/accuracy'])
        for i in range(config.eval.top_k_retrievals + 1)
    ])

    metrics['eval_best/loss'] = torch.min(all_losses, dim=0)[0].tolist()
    metrics['eval_best/accuracy'] = torch.max(all_accuracies, dim=0)[0].tolist()
    metrics['eval_best/n_tokens'] = metrics['eval_0/n_tokens']
    metrics['eval_best/problem_type'] = metrics['eval_0/problem_type']

    # Log the average accuracy with no conext, with the first context, and with the best context
    # averaged over all unique problems

    final_metrics = {}

    baseline_metrics = calculate_weighted_metrics('eval_0', metrics)
    final_metrics = {f'baseline/{k}': v for k, v in baseline_metrics.items()}

    rag_metrics = calculate_weighted_metrics('eval_1', metrics)
    final_metrics.update({f'rag/{k}': v for k, v in rag_metrics.items()})

    best_rag_metrics = calculate_weighted_metrics('eval_best', metrics)
    final_metrics.update({f'best_rag/{k}': v for k, v in best_rag_metrics.items()})

    # Log the percentage improvement from baseline -> rag, and rag -> best rag

    for metric_name, sign in [('loss', -1), ('accuracy', 1)]:
        baseline = final_metrics[f'baseline/{metric_name}']
        rag = final_metrics[f'rag/{metric_name}']
        best_rag = final_metrics[f'best_rag/{metric_name}']

        if baseline == 0:
            final_metrics[f'rag_improvement/{metric_name}'] = float('inf')
        else:
            final_metrics[f'rag_improvement/{metric_name}'] = sign * (rag - baseline) / baseline

        if rag == 0:
            final_metrics[f'best_rag_improvement/{metric_name}'] = float('inf')
        else:
            final_metrics[f'best_rag_improvement/{metric_name}'] = sign * (best_rag - rag) / rag

    return final_metrics


def run_eval(config: DictConfig, model_provider: ModelProvider, vs_provider: VectorStoreProvider):
    """Run evaluation steps for finetuning."""
    # Make a random seed that will be used for both pre- and post-finetune eval
    # This is required to keep the FIM examples the same for both runs
    eval_seed = random.randint(0, 2**32 - 1)

    # Create a directory to save results to
    output_dir = config['eval'].get('metrics_output_dir')
    now = datetime.datetime.now()
    metrics_save_dir = os.path.join(output_dir, now.strftime('%Y-%m-%d_%H-%M-%S'))

    # Reinit a new model
    model, tokenizer = create_new_model_tuple(model_provider)

    # Load name of all eval tasks
    task_metrics = []
    for task_name, task_info in load_task_info():
        logger.info(f"===== Starting eval on task [{task_name}] =====")


        ### Create and populate a vector store ###

        vector_store, n_documents = prepare_vector_store(config, vs_provider, task_info['train'])
        logger.info(f"<Task {task_name}> Added {n_documents} document nodes to vector store.")


        ### Run eval ###

        logger.info(f"Running pre-finetune eval...")
        set_seed(eval_seed)
        eval_metrics = run_task_eval(config, task_info, model, tokenizer, vector_store)
        eval_metrics['task_name'] = task_name
        eval_metrics.update(task_info['metadata'])
        task_metrics.append(eval_metrics)
        logger.debug(f"Metrics for task {task_name}: {eval_metrics}")


        ### Save metrics so far ###
        
        save_metrics(EvalResults(task_metrics=task_metrics), config, logger, output_dir=metrics_save_dir)

    return task_metrics


@hydra.main(version_base=None, config_path='../src/conf', config_name='eval')
def main(config: DictConfig):

    # Start logging and config
    configure_logging(logger)
    config = OmegaConf.create(config)
    logger.info(f"Loaded config: {config}")

    # Initialize singletons
    config_handler.ConfigProvider.initialize(config)
    model_provider = ModelProvider.get_instance(config.model)
    vs_provider = VectorStoreProvider.get_instance(config.rag)

    # Set random seed
    set_seed(config.get('seed'))
    
    # Run eval
    logger.info("Running eval...")
    run_eval(config, model_provider, vs_provider)


if __name__ == '__main__':
    main()
