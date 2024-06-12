"""Code to run HumanEval and MBPP benchmarks."""

import os

from human_eval.data import stream_jsonl, read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness
from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.routers.generator import batch_generate_completions, GenerateData


BENCHMARK_OUTPUT_DIR = '.benchmark_outputs/'
N_WORKERS = 4


def run_human_eval_benchmark(config: DictConfig, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    """Run the HumanEval benchmark.
    
    This follows the example given in the README here: https://github.com/openai/human-eval
    """

    problems = read_problems()

    n_samples_per_task = 1

    problem_tuples = [(k, v['prompt']) for k, v in problems.items()]
    task_ids, prompts = zip(*problem_tuples)

    # Create lists of the input task ids and corresponding GenerateData objects as inputs
    input_tasks = [
        task_id
        for task_id in task_ids
        for _ in range(n_samples_per_task)
    ]
    inputs = [
        GenerateData(
            prior_context=prompt,
            max_decode_length=config.train.max_gen_length
        )
        for prompt in prompts
        for _ in range(n_samples_per_task)
    ]

    # Generate completions and format them for HumanEval
    completions = batch_generate_completions(
        inputs, config, model, tokenizer, batch_size=config.train.generation_batch_size,
        progress_bar=True,
    )['output_text']
    samples = [
        dict(task_id=task_id, completion=completion)
        for task_id, completion in zip(input_tasks, completions)
    ]

    os.makedirs(BENCHMARK_OUTPUT_DIR, exist_ok=True)

    # Write the results to a file
    filepath = os.path.join(BENCHMARK_OUTPUT_DIR, 'human_eval_samples.jsonl')
    write_jsonl(filepath, samples)

    evaluate_functional_correctness(filepath, k=[1], n_workers=N_WORKERS, timeout=20)

    # Read the results
    results = list(stream_jsonl(filepath + '_results.jsonl'))
    passed = [r['passed'] for r in results]
    passed_frac = sum(passed) / len(passed)

    return passed_frac
        