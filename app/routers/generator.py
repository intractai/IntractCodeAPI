from argparse import Namespace
import logging
import threading
from typing import Annotated, Optional, Union

from concurrent.futures import ThreadPoolExecutor, CancelledError
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from transformers import PreTrainedTokenizer
import torch

from src import modeling, config
from src.data_formatting import format_inference_input


logger = logging.getLogger(__name__)
router = APIRouter()


class GenerateData(BaseModel):
    """Data class for the /generate endpoint.
    
    Args:
        file_path (str): The path to the file to generate from.
        prior_context (str): The prior context to generate from.
        proceeding_context (Optional[str], optional): 
            The proceeding context to generate from. Defaults to None.
            If provided, FIM is used, otherwise next token prediction is used.
        max_decode_length (int, optional):
            The maximum length of the generated sequence. Defaults to 128.
    """
    file_path: str
    prior_context: str
    proceeding_context: Optional[str] = None
    max_decode_length: int = 256


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


def format_input(
        item: GenerateData,
        device: Union[str, torch.device],
        tokenizer: PreTrainedTokenizer,
        cfg: Namespace):
    """Format the input for the model.

    Depending on the input, the example will either be formatted as
    a next token prediction problem or a fill in the middle problem.
    
    Args:
        item (GenerateData): The input data from a REST request.
        model (Model): The model.
        tokenizer (Tokenizer): The tokenizer.
        cfg (Namespace): The config global config.
    """

    return format_inference_input(
        preceeding_text = item.prior_context,
        tokenizer = tokenizer,
        cfg = cfg,
        proceeding_text = item.proceeding_context,
        file_path = item.file_path,
        max_decode_length = item.max_decode_length,
    ).to(device)


def generate_task(item: GenerateData, cfg: Namespace):
    # Print the current thread id to show that it is different for each request
    modeling.GLOBAL_GENERATE_THREAD_ID = threading.get_ident()
    logger.info(f"Current generate task thread id: {modeling.GLOBAL_GENERATE_THREAD_ID}")

    model_provider = modeling.ModelProvider.get_instance()
    model = model_provider.get_model()
    tokenizer = model_provider.get_model_utils()['tokenizer']

    inputs = format_input(item, model.device, tokenizer, cfg)

    outputs = model.generate(
        **inputs, max_new_tokens=item.max_decode_length,
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
