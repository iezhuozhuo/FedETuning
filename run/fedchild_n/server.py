import threading
from abc import ABC

from fedlab.utils import MessageCode
from fedlab.core.server.handler import Aggregators
from fedlab.utils.serialization import SerializationTool
from trainers.BaseServer import BaseSyncServerHandler, BaseServerManager

import torch


class FedChildSyncServerHandler(BaseSyncServerHandler, ABC):
    def __init__(self, model, valid_data, trainer=None):
        super().__init__(model, valid_data, trainer=trainer)

        self.metric_log = {
            "model_type": self.model_config.model_type,
            "clients_num": self.federated_config.clients_num,
            "alpha": self.federated_config.alpha, "task": self.data_config.task_name,
            "fl_algorithm": self.federated_config.fl_algorithm,
            "logs": []
        }

        self.client_mask_buffer = []

        self.last_parameters = SerializationTool.serialize_model(self._model)

    def valid_on_server(self):

        result = self.trainer.eval_model(
            model=self._model,
            valid_dl=self.valid_data
        )

        test_metric, test_loss = result[self.metric_name], result["eval_loss"]
        if self.global_test_best_metric < test_metric:
            self.global_test_best_metric = test_metric

        self.logger.info(f"{self.data_config.task_name}-{self.model_config.model_type} "
                         f"train with client={self.federated_config.clients_num}_"
                         f"alpha={self.federated_config.alpha}_"
                         f"epoch={self.training_config.num_train_epochs}_"
                         f"seed={self.training_config.seed}_"
                         f"comm_round={self.federated_config.rounds}")

        self.logger.debug(f"{self.federated_config.fl_algorithm} Testing "
                          f"Round: {self.round}, Current {self.metric_name}: {test_metric:.3f}, "
                          f"Current Loss: {test_loss:.3f}, Best {self.metric_name}: {self.global_test_best_metric:.3f}")

        self.metric_log["logs"].append(
            {f"round_{self.round}": {
                "loss": f"{test_loss:.3f}",
                f"{self.trainer.metric.metric_name}": f"{self.global_test_best_metric:.3f}"
            }
            }
        )

    def _update_global_model(self, parames, masks):
        assert len(parames) > 0

        if len(parames) == 1:
            self.client_buffer_cache.append(parames[0].clone())
            self.client_mask_buffer.append(masks[0].clone())
        else:
            self.client_buffer_cache += parames  # serial trainer
            self.client_mask_buffer += masks

        assert len(self.client_buffer_cache) <= self.client_num_per_round

        if len(self.client_buffer_cache) == self.client_num_per_round:
            model_parameters_list = self.client_buffer_cache
            model_mask_list = self.client_mask_buffer

            self.logger.debug(
                f"Model parameters aggregation, number of aggregation elements {len(model_parameters_list)}"
            )
            # use aggregator
            serialized_parameters = sum(model_parameters_list)

            serialized_model_mask = 1/sum(model_mask_list)
            serialized_model_mask = torch.where(serialized_model_mask != float("inf"), serialized_model_mask, 0.)
            serialized_parameters *= serialized_model_mask

            serialized_model_unmask = ~serialized_model_mask.bool()
            serialized_parameters += self.last_parameters * serialized_model_unmask

            SerializationTool.deserialize_model(self._model, serialized_parameters)
            self.last_parameters = SerializationTool.serialize_model(self._model)

            self.round += 1

            self.valid_on_server()

            # reset cache cnt
            self.client_buffer_cache = []
            self.client_mask_buffer = []

            return True  # return True to end this round.
        else:
            return False

    @property
    def downlink_package(self):
        """Property for manager layer. BaseServer manager will call this property when activates clients."""
        return [self.model_parameters]


class FedChildServerManager(BaseServerManager, ABC):
    def __init__(self, network, handler):
        super().__init__(network, handler)

    def main_loop(self):
        # while self._handler.if_stop is not True:
        #     activate = threading.Thread(target=self.activate_clients)
        #     activate.start()
        #
        #     while True:
        #         sender_rank, message_code, parames = self._network.recv()
        #         sender_rank, message_code, masks = self._network.recv()
        #
        #         if message_code == MessageCode.ParameterUpdate:
        #             if self._handler._update_global_model(parames, masks):
        #                 break
        #         else:
        #             raise Exception(
        #                 "Unexpected message code {}".format(message_code))
        self._handler.valid_on_server()
