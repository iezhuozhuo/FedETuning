"""Customer Dataloader for FedETuning"""

import os
import numpy as np

from utils import registry
from data import BaseDataLoader
from data.utils import conll_convert_examples_to_features

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

    def build_dataloader(self, features, mode="train"):
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)

        if self.model_config.model_type not in ["distilbert", "roberta"]:
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        else:
            # distilbert and roberta don't have token_type_ids
            all_token_type_ids = torch.tensor([f.attention_mask for f in features], dtype=torch.long)

        if self.output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif self.output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        if self.model_config.tuning_type and "prompt" in self.model_config.tuning_type:
            all_loss_ids = torch.tensor([f.loss_ids for f in features], dtype=torch.float)
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_loss_ids)
        else:
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

        sampler = RandomSampler(dataset) if mode == "train" else SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.training_config.train_batch_size)

        return dataloader

    def _reader_examples(self, raw_data, partition_data, n_clients,
                         train_examples_num_dict, valid_examples_num_dict,
                         train_features_dict, valid_features_dict,
                         valid_fedtures_all=None, test_fedtures_all=None
                         ):

        clients_partition_data = partition_data[self.partition_name]

        # TODO multi-task examples
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

        return federated_data


@registry.register_data("ner")
class NERDataLoader(BaseDataLoader):
    def __init__(self):
        super().__init__()

        self.output_mode = self.attribute["output_mode"]
        self.label_list = self.attribute["label_list"]

        self._load_data()

    def _reader_examples(self, raw_data, partition_data, n_clients,
                         train_examples_num_dict, valid_examples_num_dict,
                         train_features_dict, valid_features_dict,
                         valid_fedtures_all=None, test_fedtures_all=None
                         ):
        clients_partition_data = partition_data[self.partition_name]

        self.logger.info("convert train examples into features ...")
        train_features_all = np.array(conll_convert_examples_to_features(
            examples=raw_data["train"], tokenizer=self.tokenizer,
            max_length=self.data_config.max_seq_length,
            label_list=self.label_list, output_mode=self.output_mode
        ))

        self.logger.info("convert valid examples into features ...")
        valid_fedtures_all = np.array(conll_convert_examples_to_features(
            examples=raw_data["valid"], tokenizer=self.tokenizer,
            max_length=self.data_config.max_seq_length,
            label_list=self.label_list, output_mode=self.output_mode
        ))

        self.logger.info("convert test examples into features ...")
        test_fedtures_all = np.array(conll_convert_examples_to_features(
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

        return federated_data


    def build_dataloader(self, features, mode="train"):
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)

        if self.model_config.model_type not in ["distilbert", "roberta"]:
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        else:
            # distilbert and roberta don't have token_type_ids
            all_token_type_ids = torch.tensor([f.attention_mask for f in features], dtype=torch.long)

        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

        # if self.output_mode == "classification":
        #     all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        # elif self.output_mode == "regression":
        #     all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        if self.model_config.tuning_type and "prompt" in self.model_config.tuning_type:
            all_loss_ids = torch.tensor([f.loss_ids for f in features], dtype=torch.float)
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_loss_ids)
        else:
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

        sampler = RandomSampler(dataset) if mode == "train" else SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.training_config.train_batch_size)

        return dataloader



