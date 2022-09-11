"""FedBERT Meets Grandpa Axe"""

from abc import ABC
from utils import registry
from trainers.FedBaseTrainer import BaseTrainer
from run.fedbga.client import FedBGAClientTrainer, FedBGAClientManager
from run.fedbga.server import FedBGASyncServerHandler, FedBGAServerManager


@registry.register_fl_algorithm("fedbga")
class FedBGATrainer(BaseTrainer, ABC):
    def __init__(self):
        super().__init__()

        self._before_training()

    def _build_server(self):
        self.handler = FedBGASyncServerHandler(
            self.model, valid_data=self.data.valid_dataloader,
            test_data=self.data.test_dataloader
        )

        self.server_manger = FedBGAServerManager(
            network=self.network,
            handler=self.handler,
        )

    def _build_client(self):

        self.client_trainer = FedBGAClientTrainer(
            model=self.model,
            train_dataset=self.data.train_dataloader_dict,
            valid_dataset=self.data.valid_dataloader_dict,
        )

        self.client_manager = FedBGAClientManager(
            trainer=self.client_trainer,
            network=self.network
        )
