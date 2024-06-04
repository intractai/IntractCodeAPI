import logging
import math
from typing import Any, Dict, List, Tuple, Union

from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import Dataset
import numpy as np
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments
from trl import IterativeSFTTrainer
import wandb

from src.training.interactive.helpers import (
    batch_execute_code_with_timeout,
    dummy_validation_func,
    extract_code,
    openai_validate_response,
)
from src.training.interactive.multi_step import llm_generate_batched, prepare_queries


HEADER_DASHES = '-' * 30


log = logging.getLogger(__name__)


def revise_batch(
        config, batch, model, tokenizer, accelerator,
        generation_kwargs, revision_idx,
    ) -> dict:
    """Perform a single revision step on a batch of data.
    
    This function will generate responses to the queries in the batch, and then
    execute the code in the responses to determine if they are correct. If the
    responses are incorrect, then the code will be revised and the process will
    be repeated.
    
    Args:
        config: The train config.
        batch: The batch of data to revise.
        model: The model to use for generating responses.
        tokenizer: The tokenizer to use for encoding and decoding text.
        accelerator: The accelerator to use for training.
        generation_kwargs: The keyword arguments to use when generating responses.
        revision_idx: The index of the revision step.

    Returns:
        A dictionary containing the next batch of data to revise, along with
        the query and response tensors, the ground-truth response tensors, and
        the correctness of the responses.        
    """

    # Construct prompts
    query_text, query_tensors = prepare_queries(
        tokenizer, batch['instruction'], batch['code'], batch.get('exec_output', None))

    # Get response from SFTModel
    torch.cuda.empty_cache()
    try:
        response_tensors = llm_generate_batched(
            model = model,
            tokenizer = tokenizer,
            query_tensors = query_tensors,
            batch_size = config.generation_batch_size,
            accelerator = accelerator,
            **generation_kwargs,
        )
    except torch.cuda.OutOfMemoryError:
        log.info(f'Out of memory. No more revisions will be attempted this batch.')
        torch.cuda.empty_cache()
        return None

    # The response tensors also contain the queries, so we need to remove them
    response_tensors = [r[len(q):] for q, r in zip(query_tensors, response_tensors)]
    response_text = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

    # Now extract code from the outputs (if it is not already just code)
    # and execute all the code in parallel
    code_blocks = [extract_code(r) for r in response_text]
    exec_outputs = batch_execute_code_with_timeout(
        code_blocks, config.execution_timeout, config.conda_env)
    exec_outputs = [o[1][-config.execution_length_limit:] for o in exec_outputs]

    ### Compute reward score ###

    if accelerator.is_main_process:
        log.info(f"Computing reward score for revision {revision_idx}")

    critic_inputs = {
        'query': query_text,
        'response': response_text,
        'exec_output': exec_outputs,
    }
    with accelerator.split_between_processes(critic_inputs) as critic_batch:
        critic_func = openai_validate_response if config.use_critic else dummy_validation_func
        process_correct = list(map(critic_func,
            critic_batch['query'],
            critic_batch['response'],
            critic_batch['exec_output'],
        ))
        # process_correct = ([False, True] * len(critic_batch['query']))[:len(critic_batch['query'])]
    correct = gather_object(process_correct)

    if accelerator.is_main_process:
        log.info(f"Finished! {sum(correct)}/{len(correct)} correct responses.")

    ### Select incorrect responses for revision ###

    # Filter out the correct responses, leaving incorrect ones for a revision step
    full_batch = {
        'instruction': batch['instruction'],
        'code': code_blocks,
        'exec_output': exec_outputs,
        # original_idx is idx in the batch and problem_id is the idx in the whole dataset
        'original_idx': batch['original_idx'] if 'original_idx' in batch \
            else list(range(len(batch['instruction']))),
        'problem_id': batch['problem_id'],
    }

    filtered_batch = {}
    for key in full_batch:
        filtered_batch[key] = [full_batch[key][i] for i, c in enumerate(correct) if not c]

    return dict(
        next_batch = filtered_batch,
        query_tensors = query_tensors,
        response_tensors = response_tensors,
        correct = correct,
        original_idx = full_batch['original_idx'],
        problem_id = full_batch['problem_id'],
    )


def log_revision_metrics(revisions, config, tokenizer, step=None):
    """Log metrics for the revisions that were performed.

    Logged metrics:
        - fraction_correct: The fraction of responses that were correct.
        - fraction_solved: The fraction of problems that were solved.
        - correct_after: A histogram of how many revision iterations it took to find a solution.

    Logged samples:
        - r0_query, r0_response, r0_correct: The query, response, and correctness of the response
            before any revisions.
        - r1_query, r1_response, r1_correct: The query, response, and correctness of the response
            after the first revision step.
        - ...
        - ground_truth_response: The ground-truth response for each sample.
    
    Args:
        revisions: A list of dictionaries, each containing the results of a revision step.
        config: The configuration object for the training run.
        tokenizer: The tokenizer to use for encoding and decoding text.
        step: The current step in the training loop.
    """

    ### Compute metrics ###

    n_problems = len(revisions[0]['correct'])
    stats = {
        'n_samples': 0, # How many samples were processed (including revisions)
        'n_solved': 0, # How many problems were solved
        'correct_after': [], # When the correct response was found (if at all)
    }

    for i, revision_data in enumerate(revisions):
        stats['n_samples'] += len(revision_data['correct'])
        n_correct = sum(revision_data['correct'])
        stats['n_solved'] += n_correct
        stats['correct_after'].extend([i] * n_correct)

    metrics = {
        'train/fraction_correct': stats['n_solved'] / stats['n_samples'],
        'train/fraction_solved': stats['n_solved'] / n_problems,
        'train/correct_after': wandb.Histogram(stats['correct_after'], num_bins=len(revisions)),
    }
    

    ### Log samples ###

    # Decide how many samples to log, a fraction of the total number of problems
    log_frac = config.get('sample_log_frac', 0)
    n_log = np.random.binomial(n_problems, log_frac)

    if n_log > 0:
        sample_idxs = set(np.random.choice(n_problems, n_log, replace=False))
        sample_data = {idx: [] for idx in sample_idxs}

        for revision_data in revisions:

            found_idxs = set() # Keep track of which samples have been found

            for i in range(len(revision_data['correct'])):

                original_idx = revision_data['original_idx'][i]
                found_idxs.add(original_idx)

                if original_idx in sample_idxs:
                    sample_data[original_idx].extend([
                        tokenizer.decode(
                            revision_data['query_tensors'][i], skip_special_tokens=True),
                        tokenizer.decode(
                            revision_data['response_tensors'][i], skip_special_tokens=True),
                        revision_data['correct'][i],
                    ])
            
            # Fill in empty data for samples that were not found
            for sample_idx in sample_idxs - found_idxs:
                sample_data[sample_idx].extend([None, None, None])

        # Create a table for logging samples
        columns = []
        for i in range(len(revisions)):
            columns.extend([f'r{i}_query', f'r{i}_response', f'r{i}_correct'])
        sample_table = wandb.Table(columns=columns, data=list(sample_data.values()))
        metrics['train/samples'] = sample_table
        
    log.warning("Metrics are being calculated but not logged!")
    # wandb.log(metrics, step=step)


def generate_and_train_loop(
        config: DictConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataloader: torch.utils.data.DataLoader,
        accelerator: Accelerator,
        generation_kwargs: dict[str, Any],
        trainer: IterativeSFTTrainer = None,
    ) -> Dict[int, str]:
    """Train the model using the multi-step SFT algorithm.
    
    Training proceeds in a loop, where each iteration involves generating responses
    to the queries in the batch, executing the code in the responses, and revising the
    responses if they are incorrect. The loop continues until the maximum number of
    training steps or epochs is reached. The function returns a dictionary mapping
    problem IDs to their solutions. Not every problem will necessarily have a solution.

    If trainer is None, then the model only generates solutions and does not train.
    """

    data_iterator = iter(dataloader)
    max_seq_len = config.max_query_length + config.max_gen_length
    solutions = {} # Maps problem id to solution

    num_train_epochs = config.iterative_sft_trainer.num_train_epochs \
        if trainer is None \
        else trainer.args.num_train_epochs

    # Start the training loop
    epoch_idx = 0
    while trainer is None or trainer.state.global_step < trainer.args.max_steps:

        ### Prepare a batch for generation ###

        try:
            batch = next(data_iterator)
        except StopIteration:
            log.info(f"Finished epoch {epoch_idx}")
            epoch_idx += 1

            if epoch_idx >= num_train_epochs:
                break

            data_iterator = iter(dataloader)
            batch = next(data_iterator)


        ### Iterate over solutions ###

        revisions = []
        for revision_idx in range(config.max_revision_steps):

            revision_data = revise_batch(
                config, batch, model, tokenizer, accelerator,
                generation_kwargs, revision_idx,
            )

            # If there was an out-of-memory error, then we need to skip the rest of the batch
            if revision_data == None:
                break
            
            batch = revision_data.pop('next_batch')
            revisions.append(revision_data)

            # If there are no incorrect responses, then we don't need to do revisions
            if len(batch['instruction']) == 0:
                break

        
        ### Calculate and log metrics ###
        
        if accelerator.is_main_process and trainer is not None:
            log_revision_metrics(revisions, config, tokenizer, step=trainer.state.global_step)


        ### Gather correct solutions ###

        # First we need to find the correct solutions for each query
        correct_responses = {}
        for revision_data in revisions:
            for i, is_correct in enumerate(revision_data['correct']):
                if is_correct:
                    original_idx = revision_data['original_idx'][i]
                    problem_id = revision_data['problem_id'][i]
                    assert original_idx not in correct_responses, \
                        f"Multiple correct responses for query {original_idx}!"
                    correct_responses[original_idx] = revision_data['response_tensors'][i]

                    # Also store the solution text to return later
                    solutions[problem_id] = tokenizer.decode(
                        revision_data['response_tensors'][i], skip_special_tokens=True)
        
        # Stop here if not training
        if trainer is None:
            continue
        

        ### Prepare the revision data for training ###

        # Now we need to pull out the query and response training pairs form the revisions
        train_query_tensors = []
        train_gt_response_tensors = []

        # Now we can use the correct responses as the ground-truth responses
        for revision_data in revisions:
            for i in range(len(revision_data['query_tensors'])):
                original_idx = revision_data['original_idx'][i]
                if original_idx in correct_responses:
                    train_query_tensors.append(revision_data['query_tensors'][i])
                    train_gt_response_tensors.append(correct_responses[original_idx])


        ### Transform gathered data into trainer format ###

        # Input ids = query + gt_response
        # And labels = 
        #   gt_response, if mask_prompt_labels is True
        #   query + gt_response, otherwise

        # Construct inputs
        input_ids = [torch.cat([q, r])[:max_seq_len] 
                     for q, r in zip(train_query_tensors, train_gt_response_tensors)]
        attention_masks = [torch.ones_like(x) for x in input_ids]

        if config.mask_prompt_labels:
            labels = [torch.cat([torch.LongTensor([-100] * len(q)), r])[:max_seq_len] \
                        for q, r in zip(train_query_tensors, train_gt_response_tensors)]
        else:
            labels = input_ids
        
        train_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': labels,
        }


        ### Train with the formatted data ###

        torch.cuda.empty_cache()
        try:
            trainer.step(**train_inputs)
        except torch.cuda.OutOfMemoryError:
            log.error(f'Out of memory at step train step {trainer.state.global_step}. '
                       'The rest of the batch will be skipped.')
            torch.cuda.empty_cache()


        ### Cleanup ###

        del train_query_tensors
        del train_gt_response_tensors

    return solutions


def prepare_problem_dataset_and_loader(
        problem_data: Union[Dict[str, List[str]], List[str]],
        batch_size: int,
    ) -> Tuple[Dataset, DataLoader]:
    """Prepare a dataset and dataloader for a problem dataset.
    
    Args:
        problem_data (Union[Dict[str, List[str]], List[str]]): The problem data to prepare.
            Must contain the 'instruction' field, and may contain the 'code' field.
        batch_size (int): The batch size to use for the dataloader.
        
    Returns:
        Tuple[Dataset, DataLoader]: The dataset and dataloader for the problem data.
    """
    if isinstance(problem_data, list):
        problem_data = {'instruction': problem_data}

    assert 'instruction' in problem_data, \
        "The training data must contain 'instruction' field!"
    
    # Current implementation requires starting code, so just give blank code if not provided
    if 'code' not in problem_data:
        problem_data['code'] = [''] * len(problem_data['instruction'])

    # Give each problem an index to make it easy to match problems to solutions later
    problem_data['problem_id'] = list(range(len(problem_data['instruction'])))
    dataset = Dataset.from_dict(problem_data)

    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True,
        drop_last = False,
    )

    return dataset, dataloader


def train_multi_step_sft_with_verification(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_data: Union[Dict[str, List[str]], List[str]], # instruction, code (optional)
        config: DictConfig,
        **kwargs,
    ):
    """Train a model using the multi-step SFT algorithm with verification.
    
    Args:
        model (PreTrainedModel): The model to train.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding and decoding text.
        config (DictConfig): The configuration object for the training run.
        train_data Union[Dict[str, List[str]], List[str]]: The training data to use.
            Must contain the 'instruction' field, and may contain the 'code' field.
        **kwargs: Additional keyword arguments to pass to the training loop.
    """
    if torch.distributed.is_initialized():
        num_processes = torch.distributed.get_world_size()
    else:
        num_processes = 1
    

    ### Create the dataset ###

    batch_size = config.generation_batch_size * num_processes
    dataset, dataloader = prepare_problem_dataset_and_loader(train_data, batch_size)
    log.info(f"Prepared dataset with {len(dataset)} samples")


    ### Initialize the SFT trainer ###

    assert 'num_train_epochs' in config.iterative_sft_trainer, \
        "The iterative SFT trainer config must contain 'num_train_epochs' field!"
    trainer_args = TrainingArguments(**{**config.iterative_sft_trainer, **kwargs})

    # Calculate the maximum number of training steps that may be needed
    samples_per_train_step = trainer_args.per_device_train_batch_size * \
        trainer_args.gradient_accumulation_steps * \
        num_processes # Number of devices
    
    # In reality there will be less samples because not every problem will be solved,
    # and unsolved problems are not trained on, but this should be the upper bound.
    # And an upper bound is required so that the iterative trainer can set a learning
    # rate schedule.
    max_train_samples = len(dataset) * config.max_revision_steps * trainer_args.num_train_epochs
    max_train_steps = math.ceil(max_train_samples / samples_per_train_step)

    trainer_args.max_steps = max(trainer_args.max_steps, max_train_steps)

    trainer = IterativeSFTTrainer(
        model = model,
        tokenizer = tokenizer,
        args = trainer_args,
    )
    accelerator = trainer.accelerator

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None \
        else tokenizer.eos_token_id
    generation_kwargs = {
        'min_length': -1, # don't ignore the EOS token (see above)
        'top_k': 0.0, # no top-k sampling
        'top_p': 1.0, # no nucleus sampling
        'do_sample': True,
        'pad_token_id': pad_token_id,
        'max_new_tokens': config.max_gen_length, # specify how many tokens you want to generate at most
    }

    # Run the training loop
    solutions = generate_and_train_loop(
        config, model, tokenizer, dataloader, accelerator,
        generation_kwargs, trainer,
    )

    log.info(f"Finished training at step {trainer.state.global_step}")

    # Turn the solutions dictionary into a list of solutions in the same
    # order as the input problems
    ordered_solutions = [None for _ in range(len(train_data['instruction']))]
    for problem_id, solution in solutions.items():
        ordered_solutions[problem_id] = solution

    log.info(f"Found solutions for {len(solutions)}/{len(train_data['instruction'])} problems")

    return ordered_solutions


def generate_solutions(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        problem_data: Union[Dict[str, List[str]], List[str]], # instruction, code (optional)
        config: DictConfig,
    ):
    """Train a model using the multi-step SFT algorithm with verification.
    
    Args:
        model (PreTrainedModel): The model to train.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding and decoding text.
        problem_data (Union[Dict[str, List[str]], List[str]]): Problems to generate solutions for.
            Must contain the 'instruction' field, and may contain the 'code' field.
        config (DictConfig): The configuration object for the training run.
        **kwargs: Additional keyword arguments to pass to the training loop.
    """
    if torch.distributed.is_initialized():
        num_processes = torch.distributed.get_world_size()
    else:
        num_processes = 1
    

    ### Create the dataset ###
    
    batch_size = config.generation_batch_size * num_processes
    dataset, dataloader = prepare_problem_dataset_and_loader(problem_data, batch_size)
    log.info(f"Preparing dataset with {len(dataset)} samples")


    ### Initialize interactive loop params ###

    accelerator = Accelerator()

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None \
        else tokenizer.eos_token_id
    generation_kwargs = {
        'min_length': -1, # don't ignore the EOS token (see above)
        'top_k': 0.0, # no top-k sampling
        'top_p': 1.0, # no nucleus sampling
        'do_sample': True,
        'pad_token_id': pad_token_id,
        'max_new_tokens': config.max_gen_length, # specify how many tokens you want to generate at most
    }

    # Run the training loop
    solutions = generate_and_train_loop(
        config, model, tokenizer, dataloader, accelerator,
        generation_kwargs, trainer=None,
    )

    # Turn the solutions dictionary into a list of solutions in the same
    # order as the input problems
    ordered_solutions = [None for _ in range(len(problem_data['instruction']))]
    for problem_id, solution in solutions.items():
        ordered_solutions[problem_id] = solution

    log.info(f"Found solutions for {len(solutions)}/{len(problem_data['instruction'])} problems")

    return ordered_solutions
