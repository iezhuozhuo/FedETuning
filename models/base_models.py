from abc import ABC, abstractmethod
from utils import registry
import torch.nn as nn


class BaseModels(nn.Module):
    def __init__(self, task_name):
        super().__init__()

        self.task_name = task_name

        config = registry.get("config")
        self.model_config = config.model_config
        self.rank = config.federated_config.rank
        self.logger = registry.get("logger")

    def _build_config(self):
        raise NotImplementedError

    def _build_model(self):
        raise NotImplementedError

    def permutate_layers(self, model):
        raise NotImplementedError

    def freezed_layers(self, model):
        raise NotImplementedError

    def _build_efftuning_model(self):
        # . addressing a module inside the backbone model using a minimal description key.
        # . provide the interface for modifying and inserting model which keeps the docs/IO the same as the module
        #   before modification.
        # . pass a pseudo input to determine the inter dimension of the delta models.
        # . freeze a part of model parameters according to key.
        ...

    def forward(self, inputs):
        raise NotImplementedError

    @property
    def bert(self):

        if self.model_config.model_type == "bert":
            return self.backbone.bert
        elif self.model_config.model_type == "roberta":
            return self.backbone.roberta
        elif self.model_config.model_type == "distilbert":
            return self.backbone.distilbert
        else:
            raise NotImplementedError
