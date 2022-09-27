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

    def _build_config(self, **kwargs):
        auto_config = AutoConfig.from_pretrained(
            self.model_config.config_name if self.model_config.config_name else self.model_config.model_name_or_path,
            num_labels=self.num_labels,
            finetuning_task=self.task_name if self.task_name else None,
            # cache_dir=self.model_config.cache_dir,
            revision=self.model_config.model_revision,
            use_auth_token=True if self.model_config.use_auth_token else None,
        )
        return auto_config

    def _build_model(self):
        backbone = self._add_base_model()

        if getattr(self.model_config, "permutation_layers", None):
            backbone = self._add_permutate_layers(backbone)

        if self.model_config.tuning_type:
            backbone = self._add_delta_model(backbone)

        return backbone

    def _add_base_model(self):
        raise NotImplementedError

    def _add_permutate_layers(self, model):
        raise NotImplementedError

    def _add_delta_model(self, backbone):

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
