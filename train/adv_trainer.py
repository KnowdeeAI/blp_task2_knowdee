#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
"""
@author: your name eg: Yangst
@file: main.py
@time: 2022/11/26 21:00
@contact:  your email
@desc: "this is a template for pycharm, please setting for python script."
"""

from typing import Dict, Union, Any, Optional, Callable, List, Tuple

import torch
from datasets import Dataset
from torch import nn
from transformers import PreTrainedModel, DataCollator, PreTrainedTokenizerBase, EvalPrediction, TrainerCallback
from transformers import (
    Trainer,
    TrainingArguments,
)


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name="word_embeddings"):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name="word_embeddings"):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class AdvTrainer(Trainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            do_adv=True
    ):
        super(AdvTrainer, self).__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init,
                                         compute_metrics, callbacks, optimizers)
        self.do_adv = do_adv
        if do_adv:
            self.fgm = FGM(self.model)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model"s documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1 or len(loss.shape) > 0:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Adversarial training
        if self.do_adv:
            self.fgm.attack()
            with self.autocast_smart_context_manager():
                loss_adv = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss_adv = loss_adv.mean()
            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                loss_adv = loss_adv / self.args.gradient_accumulation_steps
            if self.do_grad_scaling:
                self.scaler.scale(loss_adv).backward()
            else:
                loss_adv.backward()
            self.fgm.restore()

        return loss.detach()


if __name__ == '__main__':
    pass
