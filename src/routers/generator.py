from functools import partial
import logging
import math
from typing import Annotated, List, Dict, Optional, Union

from concurrent.futures import CancelledError
from fastapi import APIRouter, Depends
from llama_index.core import VectorStoreIndex
from omegaconf import DictConfig
from pydantic import BaseModel
from transformers import BatchEncoding, PreTrainedTokenizer
import torch
from tqdm import tqdm

from src import config_handler, modeling
from src.modeling.model_hub import get_inference_model
from src.rag import get_vector_store, retrieve_context
from src.training.data_formatting import format_inference_input, format_rag_query
from src.users import validate_user_session


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
    file_path: Optional[str] = None
    prior_context: str
    proceeding_context: Optional[str] = None
    max_decode_length: int = 256


@router.post('/generate')
def generate(
        item: GenerateData,
        config: Annotated[DictConfig, Depends(config_handler.get_config)],
        username: Annotated[str, Depends(partial(validate_user_session, activity='generate'))],
    ):
    # Cancel the current task if a newer one exists
    try:
        result = generate_task(item, config, username)
    except CancelledError:
        print("Canceled generation!")
        logger.info("Cancelled /generate execution for a new request.")
        result = {'error': "Cancelled by new request"}
    return result


def generate_task(item: GenerateData, config: DictConfig, username: str):
    logging.info(f"Generating text for user: {username}.")

    model = get_inference_model(username)
    # tokenizer = get_tokenizer(username)
    vector_store = get_vector_store(username)

    results = model.generate_completion(item, config, vector_store)
    output_text = results['output_text']

    score = results.get('perplexity')
    if not math.isfinite(score):
        score = None

    return {'generated_text': output_text, 'score': score}


def batch_generate_completions(
        items: List[GenerateData],
        config: DictConfig,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        progress_bar: bool = False,
    ) -> dict:

    # TODO: This could be more efficient by batch encoding the inputs,
    # but it would require a fair bit of refactoring
    all_inputs = []
    for item in items:
        inputs = format_input(item, 'cpu', tokenizer, config)
        all_inputs.append(inputs)

    # In case we have fewer examples than batch_size
    batch_size = min(len(all_inputs), batch_size)

    if progress_bar:
        iterator = tqdm(range(0, len(all_inputs), batch_size), desc="Generating completions")
    else:
        iterator = range(0, len(all_inputs), batch_size)
    
    all_outputs = []

    for i in iterator:
        # Prevent overflow if query tensors are not even multiple of batch_size
        end_index = min(len(all_inputs), i + batch_size)
        
        batch_inputs = all_inputs[i:end_index]
        batch_inputs = {
            k: [inp[k][0] for inp in batch_inputs]
            for k in batch_inputs[0]
        }

        # Pad batch inputs to the same length
        padding_side_default = tokenizer.padding_side
        try:
            # Pad the inputs to the same length
            tokenizer.padding_side = 'left'

            padded_inputs = tokenizer.pad(
                batch_inputs,
                padding = True,
                max_length = None,
                return_tensors = 'pt',
            ).to(model.device)
        finally:
            tokenizer.padding_side = padding_side_default

        generations = model.generate(
            **padded_inputs, max_new_tokens=item.max_decode_length, **GENERATION_KWARGS)

        for generation, mask in zip(generations, padded_inputs['attention_mask']):
            output = generation[(1 - mask).sum() :] # Remove padding
            output = output[(mask).sum():] # Remove prompt

            if tokenizer.eos_token_id in output:
                pad_mask = output == tokenizer.eos_token_id
                pad_start = torch.nonzero(pad_mask, as_tuple=False)[0, 0].item()
                output = output[: pad_start + 1]  # Keep the EOS token at the end

            all_outputs.append(output)

    output_text = tokenizer.batch_decode(all_outputs, skip_special_tokens=True)

    return {
        'outputs': all_outputs,
        'output_text': output_text,
    }