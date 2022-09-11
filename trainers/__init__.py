"""FedETuning's trainers registry in trainer.__init__.py -- IMPORTANT!"""

from trainers.FedBaseTrainer import BaseTrainer
from run.fedavg.trainer import FedAvgTrainer
from run.fedbga.trainer import FedBGATrainer
from run.centralized.trainer import CenClientTrainer
from run.dry_run.trainer import DryTrainer
from run.cenchild.trainer import CenChildClientTrainer


__all__ = [
    "BaseTrainer",
    "FedAvgTrainer",
    "FedBGATrainer",
    "CenClientTrainer",
    "DryTrainer",
    "CenChildClientTrainer"
]
