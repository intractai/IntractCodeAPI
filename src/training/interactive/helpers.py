import os
import subprocess
import tempfile
import threading
import time
from typing import Any, Optional, Dict
import queue

import torch


OPENAI_API_CLIENT = None


def make_clean_env():
    env = {
        'HOME': os.environ['HOME'],
        'PATH': '/bin:/usr/bin:/sbin:/usr/sbin',
        'SHELL': os.environ['SHELL'],
    }

    # Check if anaconda or miniconda directory exists
    if os.path.exists(f'{os.environ["HOME"]}/anaconda3'):
        env['PATH'] += f':{os.environ["HOME"]}/anaconda3/bin'
    elif os.path.exists(f'{os.environ["HOME"]}/miniconda3'):
        env['PATH'] += f':{os.environ["HOME"]}/miniconda3/bin'
    elif os.path.exists(f'/opt/miniconda/bin'):
        env['PATH'] += f':/opt/miniconda/bin'
    else:
        raise Exception('No anaconda or miniconda directory found')


def execute_code_with_timeout(
        code: str,
        timeout: float,
        conda_env: Optional[str] = None,
    ) -> tuple[int, str]:
    """Execute a code string with a timeout.

    If the code takes longer than the timeout to execute, it will be terminated,
    but the output collected so far will be returned.
    
    Args:
        code (str): The code to execute.
        timeout (float): The maximum seconds to allow the code to run before terminating it.
        conda_env (str, optional): The conda environment to use for execution. Defaults to None.

    Returns:
        int: The return code of the process, or -1 if it was terminated due to a timeout.
        str: The output of the process (stdout and stderr combined).
    """
    tmp_file = tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', delete=True)
    
    tmp_file.write(code)
    tmp_file.flush()

    # If a conda_env is specified, start from a clean env, and then
    # use the run_py_script.sh script to activate the environment
    # and run the Python script.
    if conda_env:
        env = make_clean_env()
        this_dir = os.path.dirname(os.path.realpath(__file__))
        cmd = [
            f'{this_dir}/run_py_script.sh',
             conda_env,
             tmp_file.name,
        ]
    # Otherwise just use the current environment and call the
    # Python script directly
    else:
        env = os.environ.copy()
        cmd = ['python', tmp_file.name]

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        universal_newlines = True,
        env = env,
    )
    output_queue = queue.Queue()

    def enqueue_output(out, queue):
        for line in iter(out.readline, ''):
            queue.put(line)
        out.close()

    thread = threading.Thread(target=enqueue_output, args=(process.stdout, output_queue))
    thread.daemon = True
    thread.start()

    start_time = time.time()
    output_lines = []
    timed_out = False

    while process.poll() is None:
        if time.time() - start_time > timeout:
            process.terminate()
            timed_out = True
            break  # Exit the loop, but continue processing output collected so far
        while not output_queue.empty():
            output_lines.append(output_queue.get_nowait())
        time.sleep(0.1)

    # Process any remaining output after the process has terminated or timed out
    while not output_queue.empty():
        output_lines.append(output_queue.get_nowait())
    tmp_file.close()

    if timed_out:
        # If there was a timeout, return a special indicator (-1 or similar) along with the output collected so far
        return -1, ''.join(output_lines)
    else:
        # If the process completed normally, return its return code and the collected output
        return process.returncode, ''.join(output_lines)


def batch_execute_code_with_timeout(
        code: list[str],
        timeout: float,
        conda_env: Optional[str] = None
    ) -> list[tuple[int, str]]:
    """Execute a list of code strings in parallel, with a timeout.

    If any of the codes takes longer than the timeout to execute, it will be terminated,
    but the output collected so far will be returned.
    
    Args:
        codes (list[str]): The code strings to execute.
        timeout (float): The maximum seconds to allow the code to run before terminating it.
        conda_env (str, optional): The conda environment to use for execution. Defaults to None.

    Returns:
        list[tuple[int, str]]: A list of tuples, where each tuple contains the return code of
            the process and the output of the process (stdout and stderr combined).
    """
    results = [None for _ in range(len(code))]
    threads = []

    def run_code(code, idx):
        results[idx] = execute_code_with_timeout(code, timeout, conda_env)

    for i, c in enumerate(code):
        t = threading.Thread(target=lambda c=c, i=i: run_code(c, i))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    return results


def extract_code(string: str):
    """Extract code blocks from a string.

    Uses very simple heuristics to try to extract a block of code when
    a string may contain more than just code.
    This is really only necessary because we are working with pretty
    limited models for now so they don't always follow instructions
    perfectly.

    Args:
        string (str): The string to extract code blocks from.

    Returns:
        str: The extracted code block.
    """
    # First try to find a block that starts and ands with ```...``` or ```python...```
    start = string.find('```python\n')
    start = start if start == -1 else start + len('```python\n')
    if not (0 <= start <= 100):
        
        start = string.find('```\n')
        start = start if start == -1 else start + len('```\n')
    if 0 <= start <= 100:
        end = string.find('\n```', start + 1)
        end = end if end != -1 else len(string)
        return string[start:end]
    
    # If that fails, just return the whole string
    return string


def get_openai_client(api_key: Optional[str] = None):
    """Get a client for the OpenAI API.

    Returns:
        OpenAI: The OpenAI client.
    """
    from openai import OpenAI
    global OPENAI_API_CLIENT

    if OPENAI_API_CLIENT is None:
        OPENAI_API_CLIENT = OpenAI(api_key=api_key)

    return OPENAI_API_CLIENT


def openai_validate_response(
        query: str,
        response: str,
        exec_output: str,
        model: str = 'gpt-3.5-turbo',
    ) -> bool:
    """Call the OpenAI API to validate whether a response is correct.

    Args:
        query (str): The query to the agent
        response (str): The response from the agent
        exec_output (str): The output of the executed code
        model (str, optional): The model to use for validation. Defaults to 'gpt-3.5-turbo'.

    Returns:
        bool: True if the response is correct, False otherwise.
    """
    
    # First form the prompt
    system_prompt = "You are a professional software developer that evaluates code."
    prompt = f"### Query:\n{query}\n\n### Response:\n```\n{response}\n```\n\n### Execution Output:\n```\n{exec_output}\n```\n\n" + \
        "Is the response code correct, and does it fulfill the query? Respond with only 'yes' or 'no'."
    
    # Then call the API
    client = get_openai_client()
    completion = client.chat.completions.create(
        model = model,
        messages= [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ],
    )

    # Parse the result
    response = completion.choices[0].message.content
    if response[:3].lower() == 'yes':
        return True
    
    return False


class Timer:
    """A simple timer class for measuring the time taken in a block."""
    def __init__(self, name='block'):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        print(f"{self.name} took {time.time() - self.start} seconds")


class SupervisedDataCollator:
    """Simple data collator for language modeling with padding."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id #.bos_token_id

    def __call__(self, features: Any) -> Dict[str, Any]:
        bsz = len(features)
        max_length = max(len(feature['input_ids']) for feature in features)
        # max_length = self.max_length

        input_ids = torch.full((bsz, max_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(bsz, max_length, dtype=torch.long)
        labels = torch.full((bsz, max_length), -100, dtype=torch.long)

        for i, feature in enumerate(features):
            tensor_input_ids = torch.tensor(feature['input_ids'], dtype=torch.long)
            input_ids[i, :len(feature['input_ids'])] = tensor_input_ids
            attention_mask[i, :len(feature['input_ids'])] = tensor_input_ids.ne(self.pad_token_id).long()
            labels[i, :len(feature['input_ids'])] = torch.tensor(feature['labels'], dtype=torch.long)

        return dict(input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels)