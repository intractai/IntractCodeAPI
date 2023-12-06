"""FastAPI server for running code completion model as a service."""

import logging
import os
import sys
from typing import Dict
import time
import threading
import asyncio

from fastapi import FastAPI
from pydantic import BaseModel
import torch

sys.path.append('../')
from src import modeling
from src.arg_handling import parse_args
from src import finetune
from concurrent.futures import ThreadPoolExecutor, CancelledError




logger = logging.getLogger("docker_agent")


def main():
    global app, env_args, tokenizer, model, local_model_dir, device, dtype, use_flash_attention

    args, env_args, _ = parse_args(env_prefixes=["FINETUNE_"])

    app = FastAPI()

    model_name = os.getenv(
        "MODEL_NAME", "deepseek-ai/deepseek-coder-1.3b-base")
    local_model_dir = os.getenv("LOCAL_MODEL_DIR", "./.model")
    logger.info("Using model %s", model_name)
    modeling.intialize_model(model_name, local_model_dir, args)

    model = modeling.GLOBAL_MODEL
    tokenizer = modeling.GLOBAL_TOKENIZER


main()


class GenerateData(BaseModel):
    input_text: str
    max_decode_length: int = 128



@app.put("/generate")
def generate(item: GenerateData):
    global current_future

    # Cancel the current future if it exists
    try: 
        with ThreadPoolExecutor() as executor:
            # Create a new future for the incoming request
            job_thread = executor.submit(generate_task, item)
            # Run the future and return the result
            result = job_thread.result()
    except CancelledError:
        logger.info("Cancelled /generate execution by a new request.") 
        result = {"error": "Cancelled by new request"}
    return result

    

def generate_task(item: GenerateData):
    #Print the current thread id to show that it is different for each request
    modeling.GLOBAL_GENERATE_THREAD_ID = threading.get_ident()
    inputs = tokenizer(item.input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, max_length=item.max_decode_length,
        return_dict_in_generate=True, output_scores=True)
    out_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]
    output_text = tokenizer.decode(out_tokens, skip_special_tokens=True)

    # Calculate score of sequence (avg probability of the tokens)
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True,
    )
    score = torch.exp(transition_scores).mean().item()
    return {"generated_text": output_text, "score": score}


class ProjectFinetuneData(BaseModel):
    project_dict: Dict[str, str]


@app.post("/finetune/project")
def finetune_project(item: ProjectFinetuneData):
    for file_name, file_code in item.project_dict.items():
        print(f">>> {file_name}\n\n{file_code}\n\n")

    finetune_args = env_args['finetune']
    finetune.train_supervised_projectdir(
        item.project_dict, output_dir=local_model_dir,
        report_to='none', **vars(finetune_args))
    
    return {"result": "success"}