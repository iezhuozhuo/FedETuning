"""BaseModel for FedETuning"""

from abc import ABC, abstractmethod
from utils import registry
from utils import get_parameter_number
import torch.nn as nn


class BaseModels(nn.Module):
    def __init__(self, task_name):
        super().__init__()

        self.task_name = task_name

        config = registry.get("config")
        self.model_config = config.model_config
        self.rank = config.federated_config.rank
        self.logger = registry.get("logger")

    def _before_training(self):
        self.auto_config = self._build_config()
        self.backbone = self._build_model()
        self.logger.debug(f"Model Type: {self.model_config.model_type}, "
                          f"Parameters: {get_parameter_number(self.backbone)}")

    def _build_config(self):
        raise NotImplementedError

    def _build_model(self):
        raise NotImplementedError

    def permutate_layers(self, model):
        raise NotImplementedError

    def freezed_layers(self, model):
        raise NotImplementedError

    def _add_delta_model(self, backbone):
        # . addressing a module inside the backbone model using a minimal description key.
        # . provide the interface for modifying and inserting model which keeps the docs/IO the same as the module
        #   before modification.
        # . pass a pseudo input to determine the inter dimension of the delta models.
        # . freeze a part of model parameters according to key.

        from opendelta import AutoDeltaConfig
        from opendelta.auto_delta import AutoDeltaModel

        delta_args = registry.get("delta_config")
        delta_config = AutoDeltaConfig.from_dict(delta_args)
        delta_model = AutoDeltaModel.from_config(delta_config, backbone_model=backbone)
        delta_model.freeze_module(set_state_dict=True)
        # delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=False)
        # self.logger.debug(backbone)
        return backbone

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
