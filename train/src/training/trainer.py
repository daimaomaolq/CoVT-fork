import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Trainer, TrainerCallback
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
    is_peft_available,
    WEIGHTS_NAME,
    TRAINING_ARGS_NAME,
    SAFE_WEIGHTS_NAME,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
)
import safetensors
from peft import PeftModel
from typing import Optional
import numpy as np
from transformers.processing_utils import ProcessorMixin
from transformers.modeling_utils import PreTrainedModel
from peft import PeftModel
from training.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3

try:
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
except ImportError:
    zero = None
    ZeroParamStatus = None


def maybe_zero_3(param, ignore_status=False, name=None):
    if hasattr(param, "ds_id"):
        if zero is None or ZeroParamStatus is None:
            raise RuntimeError(
                "This parameter is managed by DeepSpeed ZeRO, but deepspeed is not installed."
            )
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

class QwenTrainer(Trainer):

    def __init__(self, processor, *args, **kwargs):
        super(QwenTrainer, self).__init__(*args, **kwargs)
        self.processor = processor

    def evaluation_loop(self, dataloader, description, prediction_loss_only = None, ignore_keys = None, metric_key_prefix = "eval"):
        print("I got it! Maybe for future usage")
        return super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)

    def _token_sequence(self, text):
        return self.processor.tokenizer(
            text,
            add_special_tokens=False,
        ).input_ids

    @staticmethod
    def _sequence_starts(tokens, pattern):
        if not pattern or len(pattern) > len(tokens):
            return []
        return [
            index
            for index in range(len(tokens) - len(pattern) + 1)
            if tokens[index : index + len(pattern)] == pattern
        ]

    def _i2e_token_weights(self, labels):
        answer_weight = float(self.args.i2e_answer_token_weight)
        format_weight = float(self.args.i2e_format_token_weight)
        weights = torch.ones_like(labels, dtype=torch.float32)
        if answer_weight == 1.0 and format_weight == 1.0:
            return weights

        tag_patterns = [
            self._token_sequence(tag)
            for tag in (
                "<think>",
                "</think>",
                "<explicit>",
                "</explicit>",
                "<answer>",
                "</answer>",
            )
        ]
        answer_start = tag_patterns[4]
        answer_end = tag_patterns[5]
        for row_index in range(labels.shape[0]):
            tokens = labels[row_index].tolist()
            starts = self._sequence_starts(tokens, answer_start)
            if starts:
                start = starts[0]
                ends = [
                    index
                    for index in self._sequence_starts(tokens, answer_end)
                    if index >= start
                ]
                end = ends[0] + len(answer_end) if ends else len(tokens)
                weights[row_index, start:end] = answer_weight
            if format_weight != 1.0:
                for pattern in tag_patterns:
                    for start in self._sequence_starts(tokens, pattern):
                        end = start + len(pattern)
                        weights[row_index, start:end] = torch.clamp_min(
                            weights[row_index, start:end],
                            format_weight,
                        )
        return weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        if (
            float(self.args.i2e_answer_token_weight) == 1.0
            and float(self.args.i2e_format_token_weight) == 1.0
        ):
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("Weighted I2E loss requires labels.")
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        outputs = model(**model_inputs)
        shift_logits = outputs.logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous()
        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shift_labels)
        valid = shift_labels.ne(-100)
        weights = self._i2e_token_weights(shift_labels).to(token_losses.device)
        weighted_valid = weights * valid.to(weights.dtype)
        loss = (token_losses * weighted_valid).sum() / weighted_valid.sum().clamp_min(1.0)
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """

        if is_sagemaker_mp_enabled():
            return super().create_optimizer()


        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            projection_parameters = [name for name in decay_parameters if ("_projection" in name or "query_vectors" in name or "cross_attention" in name)]
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [name for name, _ in opt_model.named_parameters() if "visual" in name and "merger" not in name]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [name for name, _ in opt_model.named_parameters() if "merger" in name]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters

                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

                if visual_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                            },
                        ]
                    )

                if merger_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                            },
                        ]
                    )
            else:
                projection_layer_lr = self.args.projection_layer_lr
                if projection_layer_lr is None:
                    projection_layer_lr = self.args.learning_rate
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and n not in projection_parameters)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and n in projection_parameters)],
                        "weight_decay": self.args.weight_decay,
                        "lr": projection_layer_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and n not in projection_parameters)],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and n in projection_parameters)],
                        "weight_decay": 0.0,
                        "lr": projection_layer_lr,
                    },
                ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        self.optimizer.param_groups
        opt = self.optimizer
        print(f"[Optimizer] {opt.__class__.__name__}")
        for i, param_group in enumerate(self.optimizer.param_groups):
            print(f"  - group {i}: lr={param_group['lr']}, "
                      f"betas={param_group.get('betas', 'N/A')}, "
                      f"eps={param_group.get('eps', 'N/A')}, "
                      f"weight_decay={param_group.get('weight_decay', 'N/A')}")
            # print(f"Param group {i}:")
            # for param in param_group["params"]:
            #     for name, p in opt_model.named_parameters():
            #         if p is param:
            #             print(f"  - {name}")
            #             break

        # self.optimizer.add_param_group({
        #                 "params": [p for n, p in opt_model.named_parameters() if ("query_vectors" in n)],
        #                 "weight_decay": self.args.weight_decay,
        #             })
        # import ipdb; ipdb.set_trace()

        return self.optimizer


    def _save_checkpoint(self, model, trial):
        if self.args.lora_enable:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            self.save_model(output_dir, _internal_call=True)

            non_lora_weights = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(), require_grad_only=False)
            torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin"))

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Save the Trainer state
            if self.args.should_save:
                # Update the `TrainerControl` state to where we are currently
                self.state.stateful_callbacks["TrainerControl"] = self.control.state()
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)

            # Maybe delete some older checkpoints.
            if self.args.should_save:
                # Solely rely on numerical checkpoint id for rotation.
                # mtime is not reliable especially on some fuse fs in cloud environments.
                self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

        else:
            super(QwenTrainer, self)._save_checkpoint(model, trial)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
            # If we are executing this function, we are the process zero, so we don't check for that.
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving model checkpoint to {output_dir}")

            supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
            # Save a trained model and configuration using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            if not isinstance(self.model, supported_classes):
                if state_dict is None:
                    state_dict = self.model.state_dict()

                if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                    self.accelerator.unwrap_model(self.model).save_pretrained(
                        output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                    )
                else:
                    logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                    if self.args.save_safetensors:
                        safetensors.torch.save_file(
                            state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                        )
                    else:
                        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
            else:
                self.model.save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )

            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)

            if self.processor is not None:
                self.processor.save_pretrained(output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    # def training_step(self, model, inputs):
    #     for name, param in model.named_parameters():
    #         if 'visual' in name and param.requires_grad:
    #             print(f"Training parameter {name}")
    #
    #     return super().training_step(model, inputs)


class UnfreezeLoRACallback(TrainerCallback):
    def __init__(self, unfreeze_step):
        self.unfreeze_step = unfreeze_step
        self.lora_lr = None
        self.trainer = None
    def set_trainer(self, trainer):
        self.trainer = trainer
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            for n, p in kwargs["model"].named_parameters():
                if "lora_" in n:
                    p.requires_grad = False
            kwargs["model"].print_trainable_parameters()
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.unfreeze_step:
            for n, p in kwargs["model"].named_parameters():
                if "lora_" in n:
                    p.requires_grad = True
            kwargs["model"].print_trainable_parameters()


class ResumeDatasetCallback(TrainerCallback):
    """
    Callback is to sync the cur_step of the Dataset when resuming training from checkpoint.

    When training is resumed from checkpoint, global_step is correctly loaded, but the cur_step of the Dataset is default to 0. This callback will detect whether the training is resumed, and if so, calculate and set the correct cur_step based on global_step.

    The formula for calculation: resumed_cur_step = global_step * batch_size * gradient_accumulation_steps
    """

    def __init__(self, train_dataset):
        self.train_dataset = train_dataset
        self._resumed = False

    def on_train_begin(self, args, state, control, **kwargs):
        if state.global_step > 0 and not self._resumed:
            samples_per_step = args.per_device_train_batch_size * args.gradient_accumulation_steps
            resumed_cur_step = state.global_step * samples_per_step

            self.train_dataset.set_cur_step(resumed_cur_step)
            self._resumed = True

            print(f"[ResumeDatasetCallback] Resumed training from global_step={state.global_step}")
            print(f"[ResumeDatasetCallback] Dataset cur_step set to {resumed_cur_step}")
            print(f"[ResumeDatasetCallback] (batch_size={args.per_device_train_batch_size}, "
                  f"grad_accum={args.gradient_accumulation_steps})")

