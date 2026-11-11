"""Training-argument helpers shared by the two training implementations.

The training flow exists twice: ``TrainingService`` (``training_service.py``,
in-process, used by the API) and ``TrainingCore`` (``training_service_core.py``,
used by the worker process). Five methods were copied word for word between
the two classes. They live here once, as a mixin both classes inherit, so the
method names and every call site stay as they were:

    _get_instruction_training_args      per-type TrainingArguments values
    _get_chat_completion_args
    _get_reasoning_training_args
    _apply_training_type_optimizations  per-type additions when Unsloth is used
    get_chat_sft_config                 SFTConfig values for chat training

Only methods whose two copies were identical were moved. Four more share a
name but not a behaviour, and stay in their own classes:

    _create_adaptive_training_args   different parameters; the worker's version
                                     ignores the training type and, with
                                     Unsloth, forces a cosine schedule and a
                                     warm-up ratio
    _detect_model_precision          different order of checks; falls back to
                                     "float16" in one class, "unknown" in the other
    _get_optimal_precision_settings  only the worker's version looks at the model
    _validate_precision_settings     the two take (training_args, model) and
                                     (model, training_args), and correct
                                     different sets of mismatches

Merging any of those would change what one of the two paths trains with, and
neither path can be run without a GPU to check.

This module imports nothing heavy, and that is deliberate. The helpers build
plain dictionaries; the one thing they need from the ML stack is the answer to
"is Unsloth available?". The two modules do not answer that the same way:
``training_service.py`` imports Unsloth under a temporarily patched ``exec``
and also counts an ``AttributeError`` as "not available", while the worker
module does a plain import. An import of Unsloth here would also run before
the patched one, because ``training_service.py`` imports this module first,
and so would undo the patch's purpose. So each class supplies its own
module's answer through ``_unsloth_available`` and nothing is decided here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:  # annotation only; the helpers read learning_rate and max_seq_length from it
    from models.schemas import FineTuneRequest


class TrainingArgsMixin:
    """Per-training-type argument dictionaries, shared by both training classes.

    A class using this mixin must define ``_unsloth_available`` and return its
    own module's ``UNSLOTH_AVAILABLE`` from it.
    """

    def _unsloth_available(self) -> bool:
        """Whether the inheriting class's module managed to import Unsloth.

        Asked on every call rather than stored, because the copies this mixin
        replaced read their module's ``UNSLOTH_AVAILABLE`` each time they ran.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must define _unsloth_available() and return its module's UNSLOTH_AVAILABLE"
        )

    def _get_instruction_training_args(
        self, adaptive_config: Dict[str, Any], request: FineTuneRequest
    ) -> Dict[str, Any]:
        """Get training arguments optimized for instruction tuning"""
        adaptive_args = adaptive_config["training_args"]
        return {
            "per_device_train_batch_size": adaptive_args["per_device_train_batch_size"],
            "gradient_accumulation_steps": adaptive_args["gradient_accumulation_steps"],
            "warmup_steps": 5,
            "learning_rate": request.learning_rate,
            "weight_decay": 0.01,
            "lr_scheduler_type": "linear",
            "logging_steps": adaptive_args.get("logging_steps", 10),
            "save_steps": adaptive_args.get("save_steps", 100),
            "dataloader_num_workers": 0,  # a single loader process: worker processes leaked semaphores
            "dataloader_pin_memory": adaptive_args.get("dataloader_pin_memory", False),
            "remove_unused_columns": False,
        }

    def _get_chat_completion_args(
        self, adaptive_config: Dict[str, Any], request: FineTuneRequest
    ) -> Dict[str, Any]:
        """Get training arguments optimized for chat completion"""
        adaptive_args = adaptive_config["training_args"]
        return {
            "per_device_train_batch_size": max(1, adaptive_args["per_device_train_batch_size"] // 2),
            "gradient_accumulation_steps": adaptive_args["gradient_accumulation_steps"] * 2,
            "warmup_steps": 10,
            "learning_rate": request.learning_rate * 0.8,
            "weight_decay": 0.005,
            "lr_scheduler_type": "cosine",
            "logging_steps": adaptive_args.get("logging_steps", 5),
            "save_steps": adaptive_args.get("save_steps", 200),
            "dataloader_num_workers": 0,  # a single loader process: worker processes leaked semaphores
            "dataloader_pin_memory": False,
            "remove_unused_columns": False,
            "group_by_length": True,
            "prediction_loss_only": False,
        }

    def _get_reasoning_training_args(
        self, adaptive_config: Dict[str, Any], request: FineTuneRequest
    ) -> Dict[str, Any]:
        """Get training arguments optimized for reasoning/GRPO training"""
        adaptive_args = adaptive_config["training_args"]
        return {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": adaptive_args["gradient_accumulation_steps"] * 4,
            "warmup_steps": 20,
            "learning_rate": request.learning_rate * 0.5,
            "weight_decay": 0.001,
            "lr_scheduler_type": "cosine",
            "logging_steps": adaptive_args.get("logging_steps", 5),
            "save_steps": adaptive_args.get("save_steps", 500),
            "dataloader_num_workers": 0,  # a single loader process: worker processes leaked semaphores
            "dataloader_pin_memory": False,
            "remove_unused_columns": False,
            "group_by_length": False,
            "prediction_loss_only": False,
        }

    def _apply_training_type_optimizations(
        self, args: Dict[str, Any], training_type: str, gpu_opts: Dict[str, Any], adaptive_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply training type specific Unsloth optimizations

        Updates ``args`` in place and returns the same dictionary. ``gpu_opts``
        is accepted and not used, as in the copies this replaced.
        """

        if not self._unsloth_available():
            return args

        if training_type == "chat":
            # "eval_accumulation_steps" was once set here too. It is not a
            # TrainingArguments parameter, so it must stay out of this dict.
            chat_optimizations = {
                "gradient_checkpointing": True,
            }

            # GRPO specific settings if reasoning is enabled
            if adaptive_config.get("enable_reasoning", False):
                chat_optimizations.update({
                    "group_by_length": False,
                    "prediction_loss_only": False,
                })

            args.update(chat_optimizations)

        elif training_type == "reasoning":
            reasoning_optimizations = {
                "gradient_checkpointing": True,
                "group_by_length": False,
                "prediction_loss_only": False,
            }
            args.update(reasoning_optimizations)

        return args

    def get_chat_sft_config(self, request: FineTuneRequest, adaptive_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get SFTConfig optimized for chat completion"""

        chat_config = adaptive_config.get("chat_config", {})

        sft_config = {
            "dataset_text_field": "text",
            "max_seq_length": min(request.max_seq_length, 4096),
            "packing": True,
            "dataset_num_proc": 1,  # a single tokenising process, for the same reason
            "chars_per_token": 3.5,
        }

        # Chat specific configurations
        if self._unsloth_available():
            sft_config.update({
                "dataset_kwargs": {
                    "add_special_tokens": False,
                    "chat_template": chat_config.get("template_type", "chatml"),
                },
                "use_unsloth_template": True,
                "assistant_only_loss": chat_config.get("assistant_only_loss", True),
            })

        return sft_config
