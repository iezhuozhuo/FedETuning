"""Customer Dataloader for FedETuning"""

import os
import numpy as np

from data import BaseDataLoader
from utils import registry, pickle_write, pickle_read, check_cached_data

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from transformers import glue_convert_examples_to_features


@registry.register_data("glue")
class GlueDataLoader(BaseDataLoader):
    def __init__(self):
        super().__init__()

        self.output_mode = self.attribute["output_mode"]
        self.label_list = self.attribute["label_list"]

        self._load_data()

    def _load_federated_data_on_server(self):
        if os.path.isfile(self.cached_data_file):
            self.logger.info(f"loading cached data ...")
            train_features_dict, valid_features_dict, valid_fedtures_all, test_fedtures_all, \
            train_examples_num_dict, valid_examples_num_dict, train_num, valid_num, test_num \
                = pickle_read(self.cached_data_file)
            # server doesn't use each client's train & test dataset
            del train_features_dict, valid_features_dict

        else:
            self.logger.info(f"generating cached data ...")
            train_features_dict, valid_features_dict, valid_fedtures_all, test_fedtures_all, \
            train_examples_num_dict, valid_examples_num_dict = self._convert_examples_to_features()

        if self.federated_config.do_mimic and self.federated_config.rank == 0:
            with open(os.path.join(self.data_config.cache_dir, "server_write.flag"), "w") as file:
                file.write("BaseServer wrote OK\n")

        self.valid_dataloader = self.build_dataloader(valid_fedtures_all, "valid")
        self.test_dataloader = self.build_dataloader(test_fedtures_all, "test")
        self.train_examples_num_dict = train_examples_num_dict
        self.valid_examples_num_dict = valid_examples_num_dict

    def _load_federated_data_on_client(self):

        train_dataloader_dict, valid_dataloader_dict = {}, {}

        if self.federated_config.do_mimic:
            self.logger.info(f"local rank {self.federated_config.rank} is waiting for processed features")
            while not check_cached_data(self.data_config.cache_dir):
                ...
            self.logger.info(f"local rank {self.federated_config.rank} builds dataloader")
            train_features_dict, valid_features_dict, valid_fedtures_all, test_fedtures_all, \
            train_examples_num_dict, valid_examples_num_dict, train_num, valid_num, test_num \
                = pickle_read(self.cached_data_file)
            del valid_fedtures_all, test_fedtures_all

            for idx in self.clients_list:
                train_dataloader_dict[idx] = self.build_dataloader(train_features_dict[idx], "train")
                valid_dataloader_dict[idx] = self.build_dataloader(valid_features_dict[idx], "valid")
        else:
            # Local data loading
            self.logger.info("Sorry, the current glue_dataloader doesn't support local loading")
            raise NotImplementedError

        self.train_dataloader_dict = train_dataloader_dict
        self.valid_dataloader_dict = valid_dataloader_dict
        self.train_examples_num_dict = train_examples_num_dict
        self.valid_examples_num_dict = valid_examples_num_dict
        self.train_num, self.valid_num, self.test_num = train_num, valid_num, test_num

    def _load_centralized_data(self):
        train_dataloader_dict, valid_dataloader_dict = {}, {}

        if os.path.isfile(self.cached_data_file):
            self.logger.info(f"loading cached data ...")
            train_features_dict, valid_features_dict, valid_fedtures_all, test_fedtures_all, \
            train_examples_num_dict, valid_examples_num_dict, train_num, valid_num, test_num \
                = pickle_read(self.cached_data_file)
        else:
            self.logger.info(f"generating cached data ...")
            train_features_dict, valid_features_dict, valid_fedtures_all, test_fedtures_all, \
            train_examples_num_dict, valid_examples_num_dict = self._convert_examples_to_features()

        train_features_all = []
        for idx, train_features in train_features_dict.items():
            train_features_all += list(train_features)

        train_dataloader_dict[-1] = self.build_dataloader(train_features_all, "train")
        valid_dataloader_dict[-1] = self.build_dataloader(valid_fedtures_all, "valid")

        self.train_dataloader_dict = train_dataloader_dict
        self.valid_dataloader_dict = valid_dataloader_dict
        self.test_dataloader = self.build_dataloader(test_fedtures_all, "test")

    def build_dataloader(self, features, mode="train"):
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if self.output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif self.output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

        sampler = RandomSampler(dataset) if mode == "train" else SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.training_config.train_batch_size)

        return dataloader

    def _convert_examples_to_features(self):

        raw_data = pickle_read(self.data_config.raw_dataset_path)
        partition_data = pickle_read(self.data_config.partition_dataset_path)

        train_examples_num_dict, valid_examples_num_dict = {}, {}
        train_features_dict, valid_features_dict = {}, {}
        clients_partition_data = partition_data[self.partition_name]

        n_clients = self.attribute["clients_num"]
        if n_clients != self.federated_config.clients_num:
            raise ValueError(f"partition data have {n_clients} clients "
                             f"that mismatches your input {self.federated_config.clients_num} clients")

        self.logger.info("convert train examples into features ...")
        train_features_all = np.array(glue_convert_examples_to_features(
            examples=raw_data["train"], tokenizer=self.tokenizer,
            max_length=self.data_config.max_seq_length,
            label_list=self.label_list, output_mode=self.output_mode
        ))

        self.logger.info("convert valid examples into features ...")
        valid_fedtures_all = np.array(glue_convert_examples_to_features(
            examples=raw_data["valid"], tokenizer=self.tokenizer,
            max_length=self.data_config.max_seq_length,
            label_list=self.label_list, output_mode=self.output_mode
        ))

        self.logger.info("convert test examples into features ...")
        test_fedtures_all = np.array(glue_convert_examples_to_features(
            examples=raw_data["test"], tokenizer=self.tokenizer,
            max_length=self.data_config.max_seq_length,
            label_list=self.label_list, output_mode=self.output_mode
        ))

        self.logger.info("build clients train & valid features ...")
        for idx in range(n_clients):
            client_train_list = clients_partition_data["train"][idx]
            train_examples_num_dict[idx] = len(client_train_list)
            train_features_dict[idx] = train_features_all[client_train_list]

            client_valid_list = clients_partition_data["valid"][idx]
            valid_examples_num_dict[idx] = len(client_valid_list)
            valid_features_dict[idx] = valid_fedtures_all[client_valid_list]

        self.train_num, self.valid_num, self.test_num = \
            len(train_features_all), len(valid_fedtures_all), len(test_fedtures_all)

        federated_data = (
            train_features_dict, valid_features_dict, valid_fedtures_all, test_fedtures_all,
            train_examples_num_dict, valid_examples_num_dict,
            self.train_num, self.valid_num, self.test_num
        )

        self.logger.info("saving processed features ...")
        pickle_write(federated_data, self.cached_data_file)

        return train_features_dict, valid_features_dict, valid_fedtures_all, test_fedtures_all, \
               train_examples_num_dict, valid_examples_num_dict,
