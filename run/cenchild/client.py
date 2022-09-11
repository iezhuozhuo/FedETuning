"""Child-network's Trainer"""

from abc import ABC

from fedlab.utils import SerializationTool
from trainers.BaseClient import BaseClientTrainer
from run.cenchild.ChildTuningOptimizer import ChildTuningAdamW

import torch
from transformers.optimization import get_linear_schedule_with_warmup


class CenChildClientTrainer(BaseClientTrainer, ABC):
    def __init__(self, model, train_dataset, valid_dataset, mode="ChildTuning-D"):
        super().__init__(model, train_dataset, valid_dataset)

        self.mode = mode

    def cen_train(self, model_parameters, gradient_mask):
        self._train_alone(
            idx=-1, model_parameters=model_parameters,
            gradient_mask=gradient_mask
        )

    def _build_optimizer(self, model, train_dl_len: int):

        if self.training_config.max_steps > 0:
            t_total = self.training_config.max_steps
            self.training_config.num_train_epochs = \
                self.training_config.max_steps // (train_dl_len // self.training_config.gradient_accumulation_steps) + 1
        else:
            t_total = \
                train_dl_len // self.training_config.gradient_accumulation_steps * self.training_config.num_train_epochs

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.backbone.named_parameters() if
                        not any(nd in n for nd in no_decay)], 'weight_decay': self.training_config.weight_decay},
            {'params': [p for n, p in model.backbone.named_parameters() if
                        any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        optimizer_cls = ChildTuningAdamW
        optimizer_kwargs = {"betas": (self.training_config.adam_beta1, self.training_config.adam_beta2),
                            "eps": self.training_config.adam_epsilon, "lr": self.training_config.learning_rate}
        optimizer = optimizer_cls(optimizer_grouped_parameters, mode=self.mode, **optimizer_kwargs)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=t_total
        )

        return optimizer, scheduler

    def _train_alone(self, idx, model_parameters, gradient_mask):
        train_loader = self._get_dataloader(dataset=self.train_dataset, client_id=idx)
        SerializationTool.deserialize_model(self._model, model_parameters)

        # build optimizer,scheduler,loss
        optimizer, scheduler = self._build_optimizer(self._model, len(train_loader))
        optimizer.set_gradient_mask(gradient_mask)
        self._model, optimizer = self._mixed_train_model(self._model, optimizer)

        for epoch in range(0, int(self.training_config.num_train_epochs)):
            self._on_epoch_begin()
            self._on_epoch(train_loader, optimizer, scheduler)
            self._on_epoch_end(idx)

    def test_on_client(self, test_dataloader):

        for idx in self.loc_best_params:
            loc_best_params = self.loc_best_params[idx]
            SerializationTool.deserialize_model(self._model, loc_best_params)
            result = self.eval.test_and_eval(
                model=self._model,
                valid_dl=test_dataloader,
                model_type=self.model_config.model_type,
                model_output_mode=self.model_config.model_output_mode
            )
            test_metric, test_loss = result[self.metric_name], result["eval_loss"]
            self.logger.critical(
                f"{self.data_config.task_name.upper()} Test, "
                f"Client:{idx}, Test loss:{test_loss:.3f}, "
                f"Test {self.metric_name}:{test_metric:.3f}"
            )
            self.loc_test_metric[idx] = test_metric
