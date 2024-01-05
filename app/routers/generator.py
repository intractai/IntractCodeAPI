import logging
from argparse import Namespace
from typing import Dict, Annotated
import threading

import torch
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, CancelledError

from src import modeling, config


logger = logging.getLogger(__name__)
router = APIRouter()


class GenerateData(BaseModel):
    input_text: str
    max_decode_length: int = 128


@router.put("/generate")
def generate(item: GenerateData, cfg: Annotated[Namespace, Depends(config.get_config)]):
    # Cancel the current future if it exists
    try: 
        with ThreadPoolExecutor() as executor:
            # Create a new future for the incoming request
            job_thread = executor.submit(generate_task, item, cfg)
            # Run the future and return the result
            result = job_thread.result()
    except CancelledError:
        print("Canceled generation!")
        logger.info("Cancelled /generate execution by a new request.") 
        result = {"error": "Cancelled by new request"}
    return result


def generate_task(item: GenerateData, cfg: Namespace):
    #Print the current thread id to show that it is different for each request
    modeling.GLOBAL_GENERATE_THREAD_ID = threading.get_ident()
    logger.info(f"Current generate task thread id: {modeling.GLOBAL_GENERATE_THREAD_ID}")
    model_provider = modeling.ModelProvider.get_instance()
    tokenizer = model_provider.get_model_utils()['tokenizer']
    model = model_provider.get_model()
    inputs = tokenizer(item.input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=128, # max_length=args.context_length,
        return_dict_in_generate=True, output_scores=True,
        do_sample=True, temperature=1.0, top_k=10, top_p=0.5)
    out_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]
    output_text = tokenizer.decode(out_tokens, skip_special_tokens=True)

    # Calculate score of sequence (avg probability of the tokens)
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True,
    )
    score = torch.exp(transition_scores).mean().item()
    return {"generated_text": output_text, "score": score}
