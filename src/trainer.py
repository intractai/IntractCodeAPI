from packaging import version
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from transformers import Trainer
from transformers.data.data_collator import DataCollator
from transformers.integrations.deepspeed import deepspeed_init
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import (
    atleast_1d,
    find_batch_size,
    IterableDatasetShard,
    nested_concat,
    nested_numpify,
)
from transformers.trainer_utils import (
    denumpify_detensorize,
    EvalPrediction,
    EvalLoopOutput,
    has_length,
    is_torch_tpu_available,
)
from transformers.training_args import TrainingArguments
from transformers.utils import is_accelerate_available, logging


if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm

if is_accelerate_available():
    from accelerate import __version__ as accelerate_version


logger = logging.get_logger(__name__)


def collate_mean_metric(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Collate a list of dictionaries with metrics into a single
    dictionary with the mean of each metric.
    """
    metrics = {}
    for key in metrics_list[0].keys():
        metrics[key] = np.mean([metric[key] for metric in metrics_list])
    return metrics


class ContinualTrainer(Trainer):

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer,
                          torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[
            torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        metrics_collator: Optional[Callable[[
            List[Dict[str, float]]], Dict[str, float]]] = None,
    ):
        super().__init__(
            model, args, data_collator, train_dataset, eval_dataset,
            tokenizer, model_init, compute_metrics, callbacks, optimizers,
            preprocess_logits_for_metrics)
        self.metrics_collator = metrics_collator

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(
            self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Storage for losses/labels/inputs on CPU
        all_losses = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0

        losses_list = []
        labels_list = []
        inputs_decode_list = []
        metrics_list = []

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            main_input_name = getattr(
                self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(
                inputs[main_input_name]) if args.include_inputs_for_metrics else None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self.accelerator.gather_for_metrics(
                    (loss.repeat(batch_size)))
                losses_list.append(losses)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(
                    labels, dim=1, pad_index=-100)
                labels = self.accelerator.gather_for_metrics((labels))
                labels_list.append(labels)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(
                    inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.accelerator.gather_for_metrics(
                    (inputs_decode))
                inputs_decode_list.append(inputs_decode)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(
                    logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.accelerator.gather_for_metrics((logits))
                # if isinstance(logits, torch.Tensor):
                #     # These take a LOT of memory, so let's move them to CPU ASAP
                #     logits = logits.cpu()

            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control)

            # Compute metrics incrementally
            if self.compute_metrics is not None and logits is not None and labels is not None:
                if args.include_inputs_for_metrics:
                    metrics = self.compute_metrics(
                        EvalPrediction(predictions=logits, label_ids=labels, inputs=inputs_decode))
                else:
                    metrics = self.compute_metrics(
                        EvalPrediction(predictions=logits, label_ids=labels))

                metrics_list.append(metrics)

            # These are too expensive to gather for each step
            logits = None

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                args.eval_accumulation_steps is not None
                and (step + 1) % args.eval_accumulation_steps == 0
                and (self.accelerator.sync_gradients or version.parse(accelerate_version) > version.parse("0.20.3"))
            ):
                # if losses_host is not None:
                #     losses = nested_numpify(losses_host)
                #     all_losses = losses if all_losses is None \
                #         else np.concatenate((all_losses, losses), axis=0)
                # if preds_host is not None:
                #     logits = nested_numpify(preds_host)
                #     all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                # if inputs_host is not None:
                #     inputs_decode = nested_numpify(inputs_host)
                #     all_inputs = (
                #         inputs_decode
                #         if all_inputs is None
                #         else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                #     )
                # if labels_host is not None:
                #     labels = nested_numpify(labels_host)
                #     all_labels = (
                #         labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                #     )

                # # Set back to None to begin a new accumulation
                # losses_host, preds_host, inputs_host, labels_host = None, None, None, None

                raise NotImplementedError(
                    'eval_accumulation_steps is not supported for ContinualTrainer')

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_list is not None:
            all_losses = torch_pad_and_cat(*losses_list)
            all_losses = nested_numpify(all_losses)
        if inputs_decode_list is not None:
            all_inputs = torch_pad_and_cat(*inputs_decode_list)
            all_inputs = nested_numpify(all_inputs)
        if labels_list is not None:
            all_labels = torch_pad_and_cat(*labels_list)
            labels = nested_numpify(all_labels)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Collate metrics
        collate_fn = self.metrics_collator or collate_mean_metric
        metrics = collate_fn(metrics_list)

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}/{key}"] = metrics.pop(key)

        return EvalLoopOutput(
            predictions=None,
            label_ids=all_labels,
            metrics=metrics,
            num_samples=num_samples
        )


def torch_pad_and_cat(*tensors, padding_index=-100):
    """Concatenates tensors on first axis, applying padding on the second if necessary."""
    tensors = [atleast_1d(x) for x in tensors]

    # Check if all tensors are already the same size
    # or that all the tensors are 1 dimensional
    if np.array([len(x.shape) == 1 for x in tensors]).all() or \
       np.array([x.shape[1] == tensors[0].shape[1] for x in tensors]).all():
        return torch.cat(tensors, dim=0)

    # Let's figure out the new shape
    new_shape = (
        sum(x.shape[0] for x in tensors),
        max(x.shape[1] for x in tensors),
    ) + tensors[0].shape[2:]

    # Now let's fill the result tensor
    result = tensors[0].new_full(new_shape, padding_index)

    dim_1_idx = 0
    for tensor in tensors:
        result[dim_1_idx:dim_1_idx+tensor.shape[0], :tensor.shape[1]] = tensor
        dim_1_idx += tensor.shape[0]

    return result
