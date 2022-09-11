"""fedbert meets grandfather axe"""

from abc import ABC
from utils import get_parameter_number
from trainers.BaseClient import BaseClientTrainer, BaseClientManager


class FedBGAClientTrainer(BaseClientTrainer, ABC):
    def __init__(self, model, train_dataset, valid_dataset):
        super().__init__(model, train_dataset, valid_dataset)

    def local_process(self, id_list, payload):
        """local process for Federated Learning"""
        model_parameters = payload[0]
        self.layer_idx = list(payload[1].numpy())
        self.param_list = self.fed_train(model_parameters, id_list)
        return self.param_list

    def get_optimized_model_params(self, model):

        modules = list()
        for layer_idx in self.model_config.server_model_layers:
            if layer_idx not in self.layer_idx:
                modules.append(model.backbone.bert.encoder.layer[int(layer_idx)])

        # .backbone.bert.encoder.layer
        # .backbone.distilbert.transformer.layer

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

        self.logger.info(get_parameter_number(model))

        optimizer_grouped_parameters = filter(
            lambda p: p.requires_grad, model.parameters()
        )
        return optimizer_grouped_parameters


class FedBGAClientManager(BaseClientManager, ABC):
    def __init__(self, network, trainer):
        super().__init__(network, trainer)
