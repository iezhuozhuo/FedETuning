"""SeqClassification Model For FedETuning """

import copy
from abc import ABC
import torch
from utils import registry
from models.base_models import BaseModels
from transformers import AutoConfig, AutoModelForSequenceClassification


@registry.register_model("seq_classification")
class SeqClassification(BaseModels, ABC):
    def __init__(self, task_name):
        super().__init__(task_name)

        self.num_labels = registry.get("num_labels")
        self.auto_config = self._build_config()
        self.backbone = self._build_model()

    def _build_config(self):
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
        backbone = AutoModelForSequenceClassification.from_pretrained(
            self.model_config.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_config.model_name_or_path),
            config=self.auto_config,
            # cache_dir=self.model_config.cache_dir,
            revision=self.model_config.model_revision,
            use_auth_token=True if self.model_config.use_auth_token else None,
            # ignore_mismatched_sizes=self.model_config.ignore_mismatched_sizes,
        )

        if getattr(self.model_config, "permutation_layers", None):
            backbone = self.permutate_layers(backbone)

        return backbone

    def permutate_layers(self, model):
        old_modules = model.bert.encoder.layer
        scrambled_modules = torch.nn.ModuleList()
        # Now iterate over all layers,
        # appending to the new module list according to the new order.
        if self.rank > 0:
            permutation = self.model_config.client_model_layers
        else:
            permutation = self.model_config.server_model_layers
        self.logger.debug(f"model's layer: {permutation}")
        for i in permutation:
            assert i <= 11, permutation
            scrambled_modules.append(old_modules[i])

        # Create a copy of the model, modify it with the new list, and return
        model_copy = copy.deepcopy(model)
        model_copy.bert.encoder.layer = scrambled_modules
        return model_copy

    def forward(self, inputs):
        output = self.backbone(**inputs)
        return output
