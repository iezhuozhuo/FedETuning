"""fedbert meets grandpa axe"""

import torch
import copy
import random
from abc import ABC
from trainers.BaseServer import BaseSyncServerHandler, BaseServerManager
from fedlab.utils import Aggregators, SerializationTool


class FedBGASyncServerHandler(BaseSyncServerHandler, ABC):
    def __init__(self, model, valid_data, test_data):
        super().__init__(model, valid_data, test_data)

        self._random_choose_layers()

    def _random_choose_layers(self):
        train_layer_num = self.model_config.train_layer_num
        all_layer_num = len(self.model_config.server_model_layers)

        part_layers = []
        if self.model_config.choose_type == "offset":
            offset = all_layer_num // train_layer_num
            for i in range(train_layer_num):
                layer_index = i*offset + random.choice([i for i in range(offset)])
                part_layers.append(layer_index)
        else:
            part_layers = random.sample([i for i in range(all_layer_num)], train_layer_num)

        self.part_layer_idx = torch.Tensor(part_layers)
        self.logger.debug(f"round: {self.round}, layer_idx: {part_layers}")

    @property
    def downlink_package(self):

        # self.client_model = copy.deepcopy(self.model)
        # old_modules = self.client_model.backbone.bert.encoder.layer
        # scrambled_modules = torch.nn.ModuleList()
        # for layer_idx in self.part_layer_idx:
        #     scrambled_modules.append(old_modules[layer_idx])
        # self.client_model.backbone.bert.encoder.layer = scrambled_modules

        return [SerializationTool.serialize_model(self.model), self.part_layer_idx]

    def _update_global_model(self, payload):
        assert len(payload) > 0

        if len(payload) == 1:
            self.client_buffer_cache.append(payload[0].clone())
        else:
            self.client_buffer_cache += payload  # serial trainer

        assert len(self.client_buffer_cache) <= self.client_num_per_round

        if len(self.client_buffer_cache) == self.client_num_per_round:
            model_parameters_list = self.client_buffer_cache
            self.logger.debug(
                f"Model parameters aggregation, number of aggregation elements {len(model_parameters_list)}"
            )

            # use aggregator
            serialized_parameters = Aggregators.fedavg_aggregate(model_parameters_list)

            # put client model into server model
            SerializationTool.deserialize_model(self.model, serialized_parameters)
            # SerializationTool.deserialize_model(self.client_model, serialized_parameters)
            # self.model_cli2ser()

            self.round += 1

            self.valid_on_server()

            # reset cache cnt
            self.client_buffer_cache = []

            # reset choose layer
            self._random_choose_layers()

            return True  # return True to end this round.
        else:
            return False

    def model_cli2ser(self):
        model_copy = copy.deepcopy(self.client_model)
        self._model.backbone.bert.embeddings = model_copy.backbone.bert.embeddings
        self._model.backbone.bert.pooler = model_copy.backbone.bert.pooler
        self._model.backbone.classifier = model_copy.backbone.classifier

        replaced_modules = model_copy.backbone.bert.encoder.layer
        origin_modules = self._model.backbone.bert.encoder.layer
        layer_num = len(self.model_config.client_model_layers)
        replace_layer_num = 0
        for layer_idx in self.part_layer_idx:
            origin_modules[layer_idx] = replaced_modules[replace_layer_num]
            replace_layer_num += 1


class FedBGAServerManager(BaseServerManager, ABC):
    def __init__(self, network, handler):
        super().__init__(network, handler)
