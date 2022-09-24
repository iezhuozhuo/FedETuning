"""BaseModel for FedETuning"""

from abc import ABC
from utils import registry
import torch.nn as nn
from transformers import trainer

from opendelta import AutoDeltaConfig
from opendelta.auto_delta import AutoDeltaModel

from models.utils import PromptType


class BaseModels(nn.Module, ABC):
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

        if any([True for PType in PromptType if PType in self.model_config.tuning_type]):
            # prefix tuning maybe in OpenDelta
            ...
        else:
            delta_args = registry.get("delta_config")
            delta_config = AutoDeltaConfig.from_dict(delta_args)
            delta_model = AutoDeltaModel.from_config(delta_config, backbone_model=backbone)
            delta_model.freeze_module(set_state_dict=True)
            # delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True)
            # self.logger.debug(delta_config)
            # self.logger.debug(backbone)
            # self.logger.debug(delta_args)

        return backbone

    def forward(self, inputs):
        raise NotImplementedError

    # @property
    # def bert(self):
    #
    #     if self.model_config.model_type == "bert":
    #         return self.backbone.bert
    #     elif self.model_config.model_type == "roberta":
    #         return self.backbone.roberta
    #     elif self.model_config.model_type == "distilbert":
    #         return self.backbone.distilbert
    #     else:
    #         raise NotImplementedError
