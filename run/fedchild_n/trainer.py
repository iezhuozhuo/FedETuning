from abc import ABC
from utils import registry, pickle_write, file_write
from trainers.FedBaseTrainer import BaseTrainer
from run.fedchild_n import FedChildClientManager, FedChildClientTrainer, LocalTrainer
from run.fedchild_n.server import FedChildSyncServerHandler, FedChildServerManager


@registry.register_fl_algorithm("fedchild_n")
class FedChildTrainer(BaseTrainer, ABC):
    def __init__(self, *args):
        super().__init__(*args)

        self._before_training()

    def _build_local_trainer(self):
        self.local_trainer = LocalTrainer()

    def _build_server(self):
        self.handler = FedChildSyncServerHandler(
            self.model, trainer=self.local_trainer,
            valid_data=self.data.valid_dataloader,
        )

        self.server_manger = FedChildServerManager(
            network=self.network,
            handler=self.handler,
        )

    def _build_client(self):
        self.client_trainer = FedChildClientTrainer(
            model=self.model,
            train_dataset=self.data.train_dataloader_dict,
            valid_dataset=self.data.valid_dataloader_dict,
            data_slices=self.federated_config.clients_id_list,
        )

        self.client_manager = FedChildClientManager(
            trainer=self.client_trainer,
            network=self.network
        )

    def train(self):

        if self.federated_config.rank == 0:
            self.logger.debug(f"Server Start ...")
            self.server_manger.run()
            pickle_write(self.handler.metric_log, self.training_config.metric_log_file)
            self.handler.metric_line += f"{self.handler.metric_name}={self.handler.global_test_best_metric:.3f}"
            file_write(self.handler.metric_line, self.training_config.metric_file, "w+")
            self.logger.info(f"watch training logs --> {self.training_config.metric_log_file}")
            self.logger.info(f"training metric --> {self.training_config.metric_file}")

        elif self.federated_config.rank > 0:
            self.logger.debug(f"Sub-Server {self.federated_config.rank} Training Start ...")
            self.client_manager.run()
        else:
            raise ValueError(f"FedChild's rank meets >= 0, but we get {self.federated_config.rank}")

