from functools import partial
import logging
from typing import Annotated, Optional, Union

from concurrent.futures import CancelledError
from fastapi import APIRouter, Depends
from omegaconf import DictConfig
from pydantic import BaseModel
from transformers import BatchEncoding, PreTrainedTokenizer
import torch

from src import config_handler, modeling
from src.modeling import get_model, get_tokenizer
from src.training.data_formatting import format_inference_input
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
    file_path: str
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


def format_input(
        item: GenerateData,
        device: Union[str, torch.device],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
    ) -> BatchEncoding:
    """Format the input for the model.

    Depending on the input, the example will either be formatted as
    a next token prediction problem or a fill in the middle problem.
    
    Args:
        item (GenerateData): The input data from a REST request.
        model (Model): The model.
        tokenizer (Tokenizer): The tokenizer.
        config (Namespace): The config global config.

    Returns:
        BatchEncoding: The formatted input with input_ids and an attention_mask.
    """
    return format_inference_input(
        preceeding_text = item.prior_context,
        tokenizer = tokenizer,
        config = config,
        proceeding_text = item.proceeding_context,
        file_path = item.file_path,
        max_decode_length = item.max_decode_length,
    ).to(device)


def generate_task(item: GenerateData, config: DictConfig, username: str):
    logging.info(f"Generating text for user: {username}.")

    model = get_model(username)
    tokenizer = get_tokenizer(username)

    inputs = format_input(item, model.device, tokenizer, config)

    outputs = model.generate(
        **inputs, max_new_tokens=item.max_decode_length,
        return_dict_in_generate=True, output_scores=True,
        do_sample=True, temperature=1.1, top_k=3)

    out_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]

    # Calculate score of sequence (avg probability of the tokens)
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True,
    )
    transition_scores = torch.exp(transition_scores[0])

    for i, s in enumerate(transition_scores):
        if s < 0.1:
            out_tokens = out_tokens[:i]
            transition_scores = transition_scores[:i]
            break

    output_text = tokenizer.decode(out_tokens, skip_special_tokens=True)
    score = transition_scores.mean().item()

    return {'generated_text': output_text, 'score': score}
