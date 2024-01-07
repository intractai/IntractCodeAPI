from argparse import Namespace
import logging
import threading
from typing import Annotated, Optional

from concurrent.futures import ThreadPoolExecutor, CancelledError
from fastapi import APIRouter, Depends
from pydantic import BaseModel
import torch
from transformers.tokenization_utils_base import BatchEncoding

from src import modeling, config
from src.modeling import FIM_BEGIN_TOKEN, FIM_HOLE_TOKEN, FIM_END_TOKEN


logger = logging.getLogger(__name__)
router = APIRouter()


# When doing FIM generation and truncation is required, this is the
# target fraction of the context that should be prior context
# (the rest is proceeding context)
PRIOR_CONTEXT_FRAC = 0.8
MAX_PREFIX_LEN = 48


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


def format_input(item: GenerateData, model, tokenizer, cfg: Namespace):
    """Format the input for the model.
    
    Args:
        item (GenerateData): The input data from a REST request.
        model (Model): The ML model.
        tokenizer (Tokenizer): The tokenizer.
        cfg (Namespace): The config global config.
    """

    # If proceeding context is not provided, use next token prediction
    if not item.proceeding_context:
        fp_tokens = tokenizer.encode(
            f'# {item.file_path}\n',
            return_tensors='pt',
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_PREFIX_LEN,
        )[0]

        max_context_length = \
            cfg.model_cfg.context_length - len(fp_tokens) - item.max_decode_length
        context_tokens = tokenizer.encode(
            item.prior_context,
            return_tensors='pt',
            add_special_tokens=False,
            truncation=True,
            max_length=max_context_length,
        )[0]

        input_ids = torch.cat([
            torch.tensor([tokenizer.bos_token_id]),
            fp_tokens,
            context_tokens,
        ]).unsqueeze(0)

        inputs = BatchEncoding({
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(tokenizer.pad_token_id).long(),
        })

    # Otherwise, do FIM
    else:
        fp_tokens = tokenizer.encode(
            f'# {item.file_path}\n',
            return_tensors='pt',
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_PREFIX_LEN,
        )[0]
        prefix = tokenizer.encode(
            item.prior_context,
            return_tensors='pt',
            add_special_tokens=False
        )[0]
        suffix = tokenizer.encode(
            item.proceeding_context,
            return_tensors='pt',
            add_special_tokens=False
        )[0]

        # Get the length of the prior and proceeding context, then truncate
        # as necessary

        # The length of the prompt + generated response cannot be longer than
        # the max context length of the model
        # -4 is for the 4 FIM and BOS special tokens added to the prompt
        max_context_length = \
            cfg.model_cfg.context_length - len(fp_tokens) - item.max_decode_length +  - 4
        raw_text_length = len(prefix) + len(suffix)

        # If the raw text is too long, truncate it
        if raw_text_length > max_context_length:
            max_prefix_length = int(max_context_length * PRIOR_CONTEXT_FRAC)
            max_suffix_length = max_context_length - max_prefix_length

            # If both the prefix and suffix are too long, truncate both
            if len(prefix) > max_prefix_length and len(suffix) > max_suffix_length:
                prefix = prefix[:max_prefix_length]
                suffix = suffix[:max_suffix_length]

            # If just the prefix is too long, truncate it
            elif len(prefix) > max_prefix_length:
                target_length = max_context_length - len(suffix)
                prefix = prefix[:target_length]

            # If just the suffix is too long, truncate it
            else:
                target_length = max_context_length - len(prefix)
                suffix = suffix[:target_length]

        prefix_tok_id = tokenizer.vocab[FIM_BEGIN_TOKEN]
        middle_tok_id = tokenizer.vocab[FIM_HOLE_TOKEN]
        suffix_tok_id = tokenizer.vocab[FIM_END_TOKEN]

        # Construct the final prompt
        input_ids = torch.cat([
            torch.tensor([tokenizer.bos_token_id]),
            torch.tensor([prefix_tok_id]), fp_tokens, prefix,
            torch.tensor([middle_tok_id]), suffix,
            torch.tensor([suffix_tok_id]),
        ]).unsqueeze(0)

        inputs = BatchEncoding({
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(tokenizer.pad_token_id).long(),
        })

    inputs = inputs.to(model.device)

    return inputs


def generate_task(item: GenerateData, cfg: Namespace):
    # Print the current thread id to show that it is different for each request
    modeling.GLOBAL_GENERATE_THREAD_ID = threading.get_ident()
    logger.info(f"Current generate task thread id: {modeling.GLOBAL_GENERATE_THREAD_ID}")

    model_provider = modeling.ModelProvider.get_instance()
    model = model_provider.get_model()
    tokenizer = model_provider.get_model_utils()['tokenizer']

    inputs = format_input(item, model, tokenizer, cfg)

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
