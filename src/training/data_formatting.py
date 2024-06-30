"""Data formatting utilities for model inference and training."""

from argparse import Namespace
import numpy as np
import logging
from typing import Optional, Sequence

from fastapi import APIRouter
from omegaconf import DictConfig
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
MAX_FP_TOKENS = 32 # Maximum tokens in file path
# Labels with -100 index are ignored in the loss calculation
# HuggingFace... document this stuff please
IGNORE_INDEX = -100
SHORT_CONTEXT_WARNING_THRESHOLD = 128


# Adapted from https://github.com/EleutherAI/gpt-neox/blob/
# FIM-clean/megatron/data/gpt2_dataset.py#L339
def prepare_fim_train_input(
        code_tokens: Sequence[int],
        fp_tokens: Sequence[int],
        tokenizer: PreTrainedTokenizer,
        truncate: bool = True):
    """
    Prepare a single training example for FIM.

    Args:
        code_tokens (Sequence[int]): The tokenized code.
        fp_tokens (Sequence[int]): The tokenized file path.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        truncate (bool, optional):
            Whether or not to truncate the input. Defaults to True.

    Returns:
        Dict[str, Any]: The formatted training example.
    """

    if truncate:
        # -6 is for -3 FIM special tokens, BOS, EOS, and 1 for extra space
        code_tokens = code_tokens[:tokenizer.model_max_length - len(fp_tokens) - 6]

    contents = tokenizer.decode(code_tokens, skip_special_tokens=False)

    try:
        # A boundary can be = 0 (prefix will be empty)
        # a boundary can be = len(contents) (suffix will be empty)
        # The two boundaries can be equal (middle will be empty)
        boundaries = list(np.random.randint(low=0, high=len(contents) + 1, size=2))
        boundaries.sort()

        # start = np.random.randint(low=0, high=len(contents))
        # mode_len = 32
        # max_len = 192
        # end = int(np.random.triangular(
        #     start + 1,
        #     start + mode_len,
        #     max(min(len(contents), max_len), start + mode_len)))
        # boundaries = [start, end]

    except ValueError as e:
        print(len(contents), contents)
        print(e)
        raise e

    prefix = contents[:boundaries[0]]
    middle = contents[boundaries[0]:boundaries[1]]
    suffix = contents[boundaries[1]:]

    prefix = np.array(tokenizer.encode(prefix, add_special_tokens=False), dtype=np.int64)
    middle = np.array(tokenizer.encode(middle, add_special_tokens=False), dtype=np.int64)
    suffix = np.array(tokenizer.encode(suffix, add_special_tokens=False), dtype=np.int64)
    fp_tokens = np.array(fp_tokens, dtype=np.int64)

    prefix_tok_id = tokenizer.vocab[FIM_BEGIN_TOKEN]
    middle_tok_id = tokenizer.vocab[FIM_HOLE_TOKEN]
    suffix_tok_id = tokenizer.vocab[FIM_END_TOKEN]

    # Here we truncate each given segment to fit the same length as it was before
    # A consequence is that we never reach the end of a file?
    # we should rather truncate at the context-level
    if truncate:
        # need to make same length as the input. Take the 3 sentinel tokens into account
        new_length = suffix.shape[0] + fp_tokens.shape[0] + prefix.shape[0] + middle.shape[0] + 3
        diff = new_length - tokenizer.model_max_length
        if diff > 0: # too long
            # If there's no space to truncate the suffix:
            # stop and report it. atm i should have stopped this from happening
            if suffix.shape[0] <= diff:
                return None
            suffix = suffix[:suffix.shape[0] - diff]

    input_ids = np.concatenate([
        [tokenizer.bos_token_id],
        [prefix_tok_id], fp_tokens, prefix,
        [middle_tok_id], suffix,
        [suffix_tok_id], middle,
        [tokenizer.eos_token_id]
    ])

    # Ignore the prompt tokens when calculating loss
    labels = np.concatenate([
        [tokenizer.bos_token_id],
        [IGNORE_INDEX] * (len(fp_tokens) + len(prefix) + 1),
        [IGNORE_INDEX] * (len(suffix) + 1),
        [IGNORE_INDEX], middle,
        [tokenizer.eos_token_id]
    ])

    return dict(
        input_ids = input_ids,
        labels = labels,
    )

def prepare_ntp_train_input(
        code_tokens: Sequence[int],
        fp_tokens: Sequence[int],
        tokenizer: PreTrainedTokenizer):
    """
    Prepare a single training example for next token prediction.

    Args:
        code_tokens (Sequence[int]): The tokenized code.
        fp_tokens (Sequence[int]): The tokenized file path.
        tokenizer (PreTrainedTokenizer): The tokenizer.

    Returns:
        Dict[str, Any]: The formatted training example.
    """
    input_ids = [tokenizer.bos_token_id] + \
        fp_tokens + \
        code_tokens + \
        [tokenizer.eos_token_id] # TODO: FIX THIS, EOS SHOULD BE PUT INTO CODE BEFORE IT IS CHUNKED

    labels = [tokenizer.bos_token_id] + \
        len(fp_tokens) * [IGNORE_INDEX] + \
        code_tokens + \
        [tokenizer.eos_token_id]


    return dict(
        input_ids = input_ids,
        labels = labels,
    )

def format_ntp_inference_input(
        text: str,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        file_path: Optional[str] = None,
        max_decode_length: int = 256,
        context_length: Optional[int] = None,
        retrieved_context: Optional[Sequence[str]] = None,
    ):
    """Format an input for next token prediction model inference.

    Args:
        text (str): The text to generate from.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        config (DictConfig): The config global config.
        file_path (Optional[str], optional):
            The path to the file to generate from. Defaults to None.
        max_decode_length (int, optional):
            The maximum length of the generated sequence. Defaults to 256.
        context_length (Optional[int], optional):
            Overrides context length from config when provided. Defaults to None.
        retrieved_context (Optional[Sequence[str]], optional):
            Additional context retrieved with the RAG retrieval system. Defaults to None.
    """

    if file_path:
        fp_tokens = tokenizer.encode(
            f'# {file_path}\n',
            return_tensors = 'pt',
            add_special_tokens = False,
            truncation = True,
            max_length = MAX_FP_TOKENS,
        )[0]
    else:
        fp_tokens = torch.tensor([], dtype=torch.long)

    if retrieved_context:
        context_template = config.rag.context_for_generation_template
        retrieved_context = [context_template.format(c) for c in retrieved_context]
        retrieved_context = '\n'.join(retrieved_context)
        retrieved_context_tokens = tokenizer.encode(
            retrieved_context,
            return_tensors = 'pt',
            add_special_tokens = False,
            truncation = True,
            max_length = config.rag.max_gen_context_length,
        )[0]
    else:
        retrieved_context_tokens = torch.tensor([], dtype=torch.long)

    context_length = context_length or config.model.context_length

    max_context_length = \
        context_length - retrieved_context_tokens - len(fp_tokens) - max_decode_length
    
    if max_context_length < 0:
        raise ValueError(
            f"The context length {context_length} is too short to fit the " + \
            f"retrieved context ({len(retrieved_context_tokens)}), file path ({len(fp_tokens)}), " + \
            f"and generated text ({max_decode_length})."
        )
    elif max_context_length <= SHORT_CONTEXT_WARNING_THRESHOLD:
        logger.warning(
            f"Only {max_context_length} tokens available for context after reserving space for the " + \
            f"retrieved context ({len(retrieved_context_tokens)}), file path ({len(fp_tokens)}), " + \
            f"and generated text ({max_decode_length}). Consider increasing the context length."
        )

    context_tokens = tokenizer.encode(
        text,
        return_tensors = 'pt',
        add_special_tokens = False,
        truncation = True,
        max_length = max_context_length,
    )[0]

    input_ids = torch.cat([
        torch.tensor([tokenizer.bos_token_id]), # TODO: FIX THIS, THERE SHOULDN'T ALWAYS BE A BOS TOKEN
        retrieved_context_tokens,
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
        config: Namespace,
        file_path: Optional[str] = None,
        max_decode_length: int = 256,
        context_length: Optional[int] = None,
        retrieved_context: Optional[Sequence[str]] = None,
    ):
    """Format an input for FIM model inference.

    Args:
        preceeding_text (str): The text preceding the hole.
        proceeding_text (str): The text following the hole.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        config (Namespace): The config global config.
        file_path (Optional[str], optional):
            The path to the file to generate from. Defaults to None.
        max_decode_length (int, optional):
            The maximum length of the generated sequence. Defaults to 256.
        context_length (Optional[int], optional):
            Overrides context length from config when provided. Defaults to None.
        retrieved_context (Optional[Sequence[str]], optional):
            Additional context retrieved with the RAG retrieval system. Defaults to None.
    """
    if file_path:
        fp_tokens = tokenizer.encode(
            f'# {file_path}\n',
            return_tensors = 'pt',
            add_special_tokens = False,
            truncation = True,
            max_length = MAX_FP_TOKENS,
        )[0]
    else:
        fp_tokens = torch.tensor([], dtype=torch.long)

    if retrieved_context:
        context_template = config.rag.context_for_generation_template
        retrieved_context = [context_template.format(c) for c in retrieved_context]
        retrieved_context = '\n'.join(retrieved_context)
        retrieved_context_tokens = tokenizer.encode(
            retrieved_context,
            return_tensors = 'pt',
            add_special_tokens = False,
            truncation = True,
            max_length = config.rag.max_gen_context_length,
        )[0]
    else:
        retrieved_context_tokens = torch.tensor([], dtype=torch.long)

    prefix = tokenizer.encode(
        preceeding_text,
        return_tensors = 'pt',
        add_special_tokens = False
    )[0]
    suffix = tokenizer.encode(
        proceeding_text,
        return_tensors = 'pt',
        add_special_tokens = False
    )[0]

    # Get the length of the prior and proceeding context, then truncate
    # as necessary

    # The length of the prompt + generated response cannot be longer than
    # the max context length of the model
    # -4 is for the 4 FIM and BOS special tokens added to the prompt
    context_length = context_length or config.model.context_length
    n_special_tokens = 4
    max_context_length = \
        context_length - len(retrieved_context_tokens) - len(fp_tokens) - max_decode_length +  - n_special_tokens
    raw_text_length = len(prefix) + len(suffix)

    if max_context_length < 0:
        raise ValueError(
            f"The context length {context_length} is too short to fit the " + \
            f"retrieved context ({len(retrieved_context_tokens)}), file path ({len(fp_tokens)}), " + \
            f"generated text ({max_decode_length}), and special tokens ({n_special_tokens})."
        )
    elif max_context_length <= SHORT_CONTEXT_WARNING_THRESHOLD:
        logger.warning(
            f"Only {max_context_length} tokens available for context after reserving space for the " + \
            f"retrieved context ({len(retrieved_context_tokens)}), file path ({len(fp_tokens)}), " + \
            f"generated text ({max_decode_length}), and special tokens ({n_special_tokens}). " + \
            "Consider increasing the context length."
        )

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
        torch.tensor([prefix_tok_id]), retrieved_context_tokens, fp_tokens, prefix,
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
        config: Namespace,
        proceeding_text: Optional[str] = None,
        file_path: Optional[str] = None,
        max_decode_length: int = 256,
        context_length: Optional[int] = None,
        retrieved_context: Optional[Sequence[str]] = None,
    ):
    """Format an input for model inference.

    Args:
        preceeding_text (str): The text preceding the hole.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        config (Namespace): The config global config.
        proceeding_text (Optional[str], optional):
            The text following the hole. Defaults to None.
        file_path (Optional[str], optional):
            The path to the file to generate from. Defaults to None.
        max_decode_length (int, optional):
            The maximum length of the generated sequence. Defaults to 256.
        context_length (Optional[int], optional):
            Overrides context length from config when provided. Defaults to None.
        retrieved_context (Optional[Sequence[str]], optional):
            The context retrieved from the vector store. Defaults to None.
    """

    if proceeding_text is None or not proceeding_text.strip():
        return format_ntp_inference_input(
            preceeding_text, tokenizer, config, file_path,
            max_decode_length, context_length, retrieved_context)
    
    return format_fim_inference_input(
        preceeding_text, proceeding_text, tokenizer, config,
        file_path, max_decode_length, context_length, retrieved_context,
    )
