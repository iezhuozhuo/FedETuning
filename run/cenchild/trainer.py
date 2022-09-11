import time
from copy import deepcopy
import numpy as np
from abc import ABC
from tqdm import tqdm

from utils import registry, file_write
from fedlab.utils import SerializationTool
from run.cenchild.client import CenChildClientTrainer
from trainers import BaseTrainer

import torch


@registry.register_fl_algorithm("cenchild")
class CenChild(BaseTrainer, ABC):
    def __init__(self, mode="ChildTuning-D", reserve_p=None):
        super().__init__()

        self.mode = mode

        if reserve_p is None:
            self.reserve_p = [0.1, 0.3, 0.5]

        self._before_training()

        self.reset_params = SerializationTool.serialize_model(self.model)

    def calculate_fisher(self, reserve_p):
        '''Calculate Fisher Information for different parameters'''
        fisher_list = registry.get("fisher_list", None)
        gradient_mask = registry.get("gradient_mask", None)

        if gradient_mask is not None:
            self.logger.info(f"Reuse the subnetwork with ChildTuning-D")

        else:
            self.logger.info(f"Find the subnetwork with ChildTuning-D")
            gradient_mask = dict()
            model = self.model
            model.to(self.training_config.device)
            model.train()

            for name, params in model.backbone.named_parameters():
                if 'layer' in name:
                    gradient_mask[params] = params.new_zeros(params.size())

            # Now begin
            N = len(self.data.train_dataloader_dict[-1])

            for step, batch in tqdm(enumerate(self.data.train_dataloader_dict[-1]), desc="FIM Training"):
                batch = tuple(t.to(self.training_config.device) for t in batch)
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]
                          }
                if self.model_config.model_type != 'distilbert':
                    # XLM, DistilBERT and RoBERTa don't use segment_ids
                    inputs['token_type_ids'] = batch[2] \
                        if self.model_config.model_type in ['bert', 'xlnet'] else None
                outputs = model(inputs)

                loss = outputs[0]

                loss.backward()

                for name, params in model.backbone.named_parameters():
                    if 'layer' in name:
                        torch.nn.utils.clip_grad_norm_(params, self.training_config.max_grad_norm)
                        gradient_mask[params] += (params.grad ** 2) / N
                model.zero_grad()

            fisher_list = None
            for k, v in gradient_mask.items():
                v = v.view(-1).cpu().numpy()
                if fisher_list is None:
                    fisher_list = v
                else:
                    fisher_list = np.append(fisher_list, v)

            registry.register("gradient_mask", gradient_mask)
            registry.register("fisher_list", fisher_list)

        self.logger.info('Calculate Fisher Information')
        polar = np.percentile(fisher_list, (1 - reserve_p) * 100)

        fisher_mask = deepcopy(gradient_mask)
        for k in gradient_mask:
            fisher_mask[k] = gradient_mask[k] >= polar
        self.logger.info('Polar => {}'.format(polar))

        # TODO: pytorch: torch.kthvalue

        return fisher_mask

    def train(self):

        for reserve_p in self.reserve_p:
            self.logger.info(f"Centralized Child-Tuning with reserve_p={reserve_p} is starting ...")

            gradient_mask = self.calculate_fisher(reserve_p)
            if gradient_mask is None:
                raise

            stime = time.time()
            self.client_trainer.cen_train(
                model_parameters=self.reset_params,
                gradient_mask=gradient_mask
            )
            cost_time = time.time()-stime

            self.on_client_end(reserve_p, cost_time)

    def on_client_end(self, reserve_p, cost_time):
        if self.training_config.do_predict:
            self.client_trainer.test_on_client(self.data.test_dataloader)

            # line, path, mode
            metric_name = self.client_trainer.metric_name
            valid_metric = self.client_trainer.loc_best_metric[-1]
            test_metric = self.client_trainer.loc_test_metric[-1]
            file_line = f"model={self.model_config.model_type}_p={reserve_p}_" \
                        f"time={cost_time:.1f}_valid_{metric_name}={valid_metric:.3f}_" \
                        f"test_{metric_name}={test_metric:.3f}"

            file_write(file_line, self.training_config.metric_file, "a+")

            self.logger.debug(f"Centralized Child-Tuning with reserve_p={reserve_p} costs {cost_time:.1f}s and "
                              f"Best Test {metric_name.upper()}={test_metric:.3f}")

            # 重置 child-network trainer
            self._build_client()

    def _build_client(self):
        self.client_trainer = CenChildClientTrainer(
            model=self.model,
            train_dataset=self.data.train_dataloader_dict,
            valid_dataset=self.data.valid_dataloader_dict,
            mode=self.mode
        )
