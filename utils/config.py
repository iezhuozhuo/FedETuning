"""config for FedETuning"""

import os
import time
from abc import ABC
from omegaconf import OmegaConf
from transformers import HfArgumentParser

from utils import make_sure_dirs, rm_file
from utils.register import registry
from configs import ModelArguments, DataTrainingArguments, TrainArguments, FederatedTrainingArguments


class Config(ABC):
    def __init__(self, model_args, data_args, training_args, federated_args):
        self.model_config = model_args
        self.data_config = data_args
        self.training_config = training_args
        self.federated_config = federated_args

    def save_configs(self):
        ...

    def check_config(self):
        self.config_check_federated()

    def config_check_federated(self):

        if "cen" in self.F.fl_algorithm:
            if self.F.rank == -1:
                self.F.world_size = 1
            else:
                raise ValueError(f"Must set world_size, but find {self.F.world_size}")
        else:
            if self.F.clients_num % (self.F.world_size - 1):
                raise ValueError(f"{self.F.clients_num} % {(self.F.world_size - 1)} != 0")

    @property
    def M(self):
        return self.model_config

    @property
    def D(self):
        return self.data_config

    @property
    def T(self):
        return self.training_config

    @property
    def F(self):
        return self.federated_config


def amend_config(model_args, data_args, training_args, federated_args):
    config = Config(model_args, data_args, training_args, federated_args)
    # load customer config (hard code)
    # TODO args in config.yaml can overwrite --arg
    root_folder = registry.get("root_folder")
    cust_config_path = os.path.join(root_folder, f"run/{config.F.fl_algorithm}/config.yaml")
    if os.path.isfile(cust_config_path):
        cust_config = OmegaConf.load(cust_config_path)
        for key, values in cust_config.items():
            if values:
                args = getattr(config, key)
                for k, v in values.items():
                    setattr(args, k, v)

    # set training path
    config.T.output_dir = os.path.join(config.T.output_dir, config.D.task_name)
    make_sure_dirs(config.T.output_dir)

    if not config.D.cache_dir:
        cache_dir = os.path.join(config.T.output_dir, "cached_data")
        if config.F.rank != -1:
            config.D.cache_dir = os.path.join(
                cache_dir, f"cached_{config.M.model_type}_{config.F.clients_num}_{config.F.alpha}"
            )
        else:
            config.D.cache_dir = os.path.join(
                cache_dir, f"cached_{config.M.model_type}_centralized"
            )
    make_sure_dirs(config.D.cache_dir)

    # set training_args
    config.T.save_dir = os.path.join(config.T.output_dir, config.F.fl_algorithm.lower())
    make_sure_dirs(config.T.save_dir)
    config.T.checkpoint_dir = os.path.join(config.T.save_dir, "saved_model")
    make_sure_dirs(config.T.checkpoint_dir)
    # set phase
    phase = "train" if config.T.do_train else "evaluate"
    registry.register("phase", phase)
    # set metric log path
    times = time.strftime("%Y%m%d%H%M%S", time.localtime())
    registry.register("run_time", times)
    config.T.times = times
    config.T.metric_file = os.path.join(config.T.save_dir, f"{config.M.model_type}.eval")
    config.T.metric_log_file = os.path.join(config.T.save_dir, f"{times}_{config.M.model_type}.eval.log")

    # set federated_args
    if config.F.do_mimic and config.F.rank == 0:
        server_write_flag_path = os.path.join(config.D.cache_dir, "server_write.flag")
        rm_file(server_write_flag_path)

    if config.F.partition_method is None:
        config.F.partition_method = f"clients={config.F.clients_num}_alpha={config.F.alpha}"

    config.check_config()
    registry.register("config", config)
    return config


def build_config():
    # read parameters
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainArguments, FederatedTrainingArguments))
    model_args, data_args, training_args, federated_args = parser.parse_args_into_dataclasses()

    # amend and register configs
    config = amend_config(model_args, data_args, training_args, federated_args)

    # logging fl & some path
    logger = registry.get("logger")
    logger.info(f"FL-Algorithm: {config.federated_config.fl_algorithm}")
    logger.info(f"output_dir: {config.training_config.output_dir}")
    logger.info(f"cache_dir: {config.data_config.cache_dir}")
    logger.info(f"save_dir: {config.training_config.save_dir}")
    logger.info(f"checkpoint_dir: {config.training_config.checkpoint_dir}")
    logger.info(f"base_info: {config.M.model_type}_num={config.F.clients_num}_"
                f"alp={config.F.alpha}_rd={config.F.rounds}_smp={config.F.sample}")

    return config
