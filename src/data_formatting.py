"""Data formatting utilities for model inference and training."""

from argparse import Namespace
import logging
from typing import Optional

from fastapi import APIRouter
import torch
from transformers.tokenization_utils_base import BatchEncoding
from transformers import PreTrainedTokenizer

from src.modeling import FIM_BEGIN_TOKEN, FIM_HOLE_TOKEN, FIM_END_TOKEN


logger = logging.getLogger(__name__)
router = APIRouter()


# When doing FIM generation and truncation is required, this is the
# target fraction of the context that should be prior context
# (the rest is proceeding context)
GEN_PRIOR_CONTEXT_FRAC = 0.8
MAX_FP_LEN = 32 # Maximum tokens in file path


def format_ntp_inference_input(
        text: str,
        tokenizer: PreTrainedTokenizer,
        cfg: Namespace,
        file_path: Optional[str] = None,
        max_decode_length: int = 256):
    """Format an input for next token prediction model inference.

    Args:
        text (str): The text to generate from.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        cfg (Namespace): The config global config.
        file_path (Optional[str], optional):
            The path to the file to generate from. Defaults to None.
        max_decode_length (int, optional):
            The maximum length of the generated sequence. Defaults to 256.
    """

    fp_tokens = tokenizer.encode(
        f'# {file_path}\n',
        return_tensors='pt',
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_FP_LEN,
    )[0]

    max_context_length = \
        cfg.model_cfg.context_length - len(fp_tokens) - max_decode_length
    context_tokens = tokenizer.encode(
        text,
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

    return inputs


def format_fim_inference_input(
        preceeding_text: str,
        proceeding_text: str,
        tokenizer: PreTrainedTokenizer,
        cfg: Namespace,
        file_path: Optional[str] = None,
        max_decode_length: int = 256):
    """Format an input for FIM model inference.

    Args:
        preceeding_text (str): The text preceding the hole.
        proceeding_text (str): The text following the hole.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        cfg (Namespace): The config global config.
        file_path (Optional[str], optional):
            The path to the file to generate from. Defaults to None.
        max_decode_length (int, optional):
            The maximum length of the generated sequence. Defaults to 256.
    """

    fp_tokens = tokenizer.encode(
        f'# {file_path}\n',
        return_tensors='pt',
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_FP_LEN,
    )[0]
    prefix = tokenizer.encode(
        preceeding_text,
        return_tensors='pt',
        add_special_tokens=False
    )[0]
    suffix = tokenizer.encode(
        proceeding_text,
        return_tensors='pt',
        add_special_tokens=False
    )[0]

    # Get the length of the prior and proceeding context, then truncate
    # as necessary

    # The length of the prompt + generated response cannot be longer than
    # the max context length of the model
    # -4 is for the 4 FIM and BOS special tokens added to the prompt
    max_context_length = \
        cfg.model_cfg.context_length - len(fp_tokens) - max_decode_length +  - 4
    raw_text_length = len(prefix) + len(suffix)

    # If the raw text is too long, truncate it
    if raw_text_length > max_context_length:
        max_prefix_length = int(max_context_length * GEN_PRIOR_CONTEXT_FRAC)
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

    return inputs

def format_inference_input(
        preceeding_text: str,
        tokenizer: PreTrainedTokenizer,
        cfg: Namespace,
        proceeding_text: Optional[str] = None,
        file_path: Optional[str] = None,
        max_decode_length: int = 256):
    """Format an input for model inference.

    Args:
        preceeding_text (str): The text preceding the hole.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        cfg (Namespace): The config global config.
        proceeding_text (Optional[str], optional):
            The text following the hole. Defaults to None.
        file_path (Optional[str], optional):
            The path to the file to generate from. Defaults to None.
        max_decode_length (int, optional):
            The maximum length of the generated sequence. Defaults to 256.
    """
    
    if proceeding_text is None or not proceeding_text.strip():
        return format_ntp_inference_input(
            preceeding_text, tokenizer, cfg, file_path, max_decode_length)
    
    return format_fim_inference_input(
        preceeding_text, proceeding_text, tokenizer, cfg, file_path, max_decode_length)
