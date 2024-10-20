from functools import partial
import logging
import math
from typing import Annotated, List, Optional, Union

from concurrent.futures import CancelledError
from fastapi import APIRouter, Depends
from llama_index.core import VectorStoreIndex
from omegaconf import DictConfig
from pydantic import BaseModel
from transformers import BatchEncoding, PreTrainedTokenizer
import torch
from tqdm import tqdm

from src import config_handler, modeling
from src.modeling import get_model, get_tokenizer
from src.rag import get_vector_store, retrieve_context
from src.training.data_formatting import format_inference_input, format_rag_query
from src.users import validate_user_session


GENERATION_KWARGS = dict(
    temperature=0.7, do_sample=True, top_k=5,
    # num_beams=3, early_stopping=True, 
    # do_sample=True, temperature=1.1, top_k=3,
)

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


def prepare_input(
        item: GenerateData,
        config: DictConfig,
        device: Union[str, torch.device],
        tokenizer: PreTrainedTokenizer,
        vector_store: Optional[VectorStoreIndex] = None,
    ) -> BatchEncoding:
    """Prepare and format input for the model.

    Depending on the input, the example will either be formatted as
    a next token prediction problem or a fill in the middle problem.
    RAG context will also be added if enabled in the config.
    
    Args:
        item (GenerateData): The input data from a REST request.
        config (DictConfig): The config global config.
        model (Model): The model.
        tokenizer (Tokenizer): The tokenizer.
        vector_store (VectorStoreIndex, optional): The vector store for RAG. Defaults to None.

    Returns:
        BatchEncoding: The formatted input with input_ids and an attention_mask.
    """
    use_rag = config.rag.get('enabled', False) and vector_store is not None
    if use_rag:
        # Create a user context string from the prior and proceeding context
        user_context_str = format_rag_query(
            prior_context = item.prior_context,
            proceeding_context = item.proceeding_context,
            max_length = config.rag.max_embed_context_length,
        )    
        
        # Then retrieve the relevant context strings from the vector store
        retrieved = retrieve_context(user_context_str, vector_store, top_k=config.rag.n_chunks_per_generation)
    else:
        retrieved = None

    context_length = config.model.context_length

    inputs = format_inference_input(
        preceeding_text = item.prior_context,
        tokenizer = tokenizer,
        config = config,
        proceeding_text = item.proceeding_context,
        file_path = item.file_path,
        max_decode_length = config.inference.max_gen_length,
        context_length = context_length,
        retrieved_context = retrieved,
    )

    return inputs.to(device)


def generate_task(item: GenerateData, config: DictConfig, username: str):
    logging.info(f"Generating text for user: {username}.")

    model = get_model(username)
    tokenizer = get_tokenizer(username)
    vector_store = get_vector_store(username)

    results = generate_completion(item, config, model, tokenizer, vector_store)
    output_text = results['output_text']

    score = results.get('perplexity').item()
    if not math.isfinite(score):
        score = None

    return {'generated_text': output_text, 'score': score}


def generate_completion(
        item: GenerateData,
        config: DictConfig,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        vector_store: Optional[VectorStoreIndex] = None,
    ) -> dict:
    inputs = prepare_input(item, config, model.device, tokenizer, vector_store)

    outputs = model.generate(
        **inputs, max_new_tokens=config.inference.max_gen_length,
        return_dict_in_generate=True, output_scores=True,
        **GENERATION_KWARGS,
    )

    out_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]
    output_text = tokenizer.decode(out_tokens, skip_special_tokens=True)
    logits = torch.stack(outputs.scores[-len(out_tokens):]).squeeze(1)
    probs = logits.softmax(dim=1).gather(1, out_tokens.unsqueeze(1))

    perplexity = torch.exp(-torch.sum(torch.log(probs)) / len(out_tokens))

    return {
        'outputs': outputs,
        'output_text': output_text,
        'perplexity': perplexity,
    }


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
        inputs = prepare_input(item, config, 'cpu', tokenizer)
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