import numpy as np
from abc import ABC
from tqdm import tqdm

import torch
from transformers import get_linear_schedule_with_warmup

from fedlab.utils import MessageCode
from trainers.BaseClient import BaseClientTrainer, BaseLocalTrainer, BaseClientManager
from fedlab.utils.serialization import SerializationTool
from run.cenchild.ChildTuningOptimizer import ChildTuningAdamW


class ChildSerializationTool(SerializationTool):

    @staticmethod
    def serialize_params_list(params_list) -> torch.Tensor:
        parameters = [param.data.view(-1) for param in params_list]
        m_parameters = torch.cat(parameters)
        m_parameters = m_parameters.cpu()

        return m_parameters

    @staticmethod
    def serialize_mask_list(model_mask: dict):
        masks = [mask.view(-1) for _, mask in model_mask.items()]
        masks = torch.cat(masks)
        return torch.as_tensor(masks.cpu().numpy().astype(float))


class LocalTrainer(BaseLocalTrainer, ABC):
    def __init__(self, mode='ChildTuning-D'):
        super().__init__()
        self.mode = mode

    def train_model(self, model, train_dl, gradient_mask):
        model.to(self.device)

        # build optimizer and scheduler
        optimizer, scheduler = self._build_optimizer(model, len(train_dl))
        optimizer.set_gradient_mask(gradient_mask)
        model, optimizer = self._mixed_train_model(model, optimizer)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0

        for epoch in range(0, int(self.training_config.num_train_epochs)):
            for step, batch in enumerate(train_dl):
                model.train()
                batch = tuple(t.to(self.device) for t in batch)
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
                # loss = criterion(logits.view(-1, model.num_labels), inputs["labels"].view(-1))

                if self.training_config.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.training_config.gradient_accumulation_steps > 1:
                    loss = loss / self.training_config.gradient_accumulation_steps

                if self.training_config.fp16:
                    try:
                        from apex import amp
                    except ImportError:
                        raise ImportError(
                            "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.training_config.gradient_accumulation_steps == 0:
                    if self.training_config.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.training_config.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.training_config.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

            self.logger.info(f"Local Epoch {epoch} is done and training loss is {tr_loss / global_step:.3f}")

        return global_step, tr_loss / global_step

    def eval_model(self, model, valid_dl):
        if self.training_config.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        results = {}
        for batch in tqdm(valid_dl, desc="Centralized Evaluating"):
            model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                if self.model_config.model_type != 'distilbert':
                    # XLM, DistilBERT and RoBERTa don't use segment_ids
                    inputs['token_type_ids'] = \
                        batch[2] if self.model_config.model_type in ['bert', 'xlnet'] else None
                outputs = model(inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results["eval_loss"] = eval_loss
        if self.model_config.model_output_mode == "seq_classification":
            preds = np.argmax(preds, axis=1)
        elif self.model_config.model_output_mode == "regression":
            preds = np.squeeze(preds)

        self.metric.update_metrics(preds, out_label_ids)
        results.update(self.metric.best_metric)
        return results

    def _build_optimizer(self, model, train_dl_len: int):

        if self.training_config.max_steps > 0:
            t_total = self.training_config.max_steps
            self.training_config.num_train_epochs = \
                self.training_config.max_steps // (train_dl_len // self.training_config.gradient_accumulation_steps) + 1
        else:
            t_total = \
                train_dl_len // self.training_config.gradient_accumulation_steps * self.training_config.num_train_epochs

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.backbone.named_parameters() if
                        not any(nd in n for nd in no_decay)], 'weight_decay': self.training_config.weight_decay},
            {'params': [p for n, p in model.backbone.named_parameters() if
                        any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        optimizer_cls = ChildTuningAdamW
        optimizer_kwargs = {"betas": (self.training_config.adam_beta1, self.training_config.adam_beta2),
                            "eps": self.training_config.adam_epsilon, "lr": self.training_config.learning_rate}
        optimizer = optimizer_cls(optimizer_grouped_parameters, mode=self.mode, **optimizer_kwargs)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=t_total
        )

        return optimizer, scheduler


class FedChildClientTrainer(BaseClientTrainer, ABC):
    def __init__(self, model, train_dataset, valid_dataset, data_slices):
        client_num = len(data_slices)
        super().__init__(model, client_num)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        # self.test_dataset = valid_dataset
        self.data_slices = data_slices  # [0, client_num)
        self.local_trainer = LocalTrainer()

        # self.local_last_model = {idx: ChildSerializationTool.serialize_model(self._model)
        #                          for idx in data_slices}

        self.masks_list = []

    def _get_dataloader(self, dataset, client_id):
        if isinstance(dataset, dict):
            data_loader = dataset[client_id]
        else:
            data_loader = dataset
        return data_loader

    def _train_alone(self, model_parameters, train_loader):
        SerializationTool.deserialize_model(self._model, model_parameters)
        gradient_mask, model_mask = self.calculate_fisher(reserve_p=1.0, train_loader=train_loader)
        self.local_trainer.train_model(model=self._model, train_dl=train_loader, gradient_mask=gradient_mask)
        return gradient_mask, model_mask

    def calculate_fisher(self, reserve_p, train_loader):
        """
        Calculate Fisher Information for different parameters
        """
        self.logger.info(f"Find the subnetwork with ChildTuning-D")

        gradient_mask = dict()
        model = self.model
        model.to(self.training_config.device)
        model.train()

        model_mask = dict()
        for name, params in model.backbone.named_parameters():
            if 'layer' in name:
                gradient_mask[params] = params.new_zeros(params.size())
                model_mask[name] = gradient_mask[params]
            else:
                model_mask[name] = params.new_ones(params.size())

        # Now begin
        N = len(train_loader)

        for step, batch in tqdm(enumerate(train_loader), desc="FIM Training"):
            batch = tuple(t.to(self.training_config.device) for t in batch)
            inputs = {
                'input_ids': batch[0],
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

        # TODO: pytorch: torch.kthvalue
        fisher_list = None
        for k, v in gradient_mask.items():
            v = v.view(-1).cpu().numpy()
            if fisher_list is None:
                fisher_list = v
            else:
                fisher_list = np.append(fisher_list, v)

        self.logger.info(f'Calculate Fisher Information with Reserve_P={reserve_p}')
        polar = np.percentile(fisher_list, (1 - reserve_p) * 100)

        for k in gradient_mask:
            gradient_mask[k] = gradient_mask[k] >= polar
        self.logger.info('Polar => {}'.format(polar))

        for name, params in model.backbone.named_parameters():
            if 'layer' in name:
                assert name in model_mask
                model_mask[name] = gradient_mask[params]

        return gradient_mask, model_mask

    def train(self, model_parameters, id_list):
        param_list = []
        masks_list = []
        for idx in id_list:
            train_data_loader = self._get_dataloader(dataset=self.train_dataset, client_id=idx)
            gradient_mask, model_mask = self._train_alone(
                model_parameters=model_parameters,
                train_loader=train_data_loader,
            )
            param_list.append(self.get_upload_parameters(gradient_mask, model_mask))
            masks_list.append(self.mask_serialized(model_mask))
        return param_list, masks_list

    def local_process(self, id_list, payload):
        model_parameters = payload[0]
        self.param_list, self.masks_list = self.train(model_parameters, id_list)
        return self.param_list, self.masks_list

    def mask_serialized(self, mask):
        """Return serialized model mask."""
        return ChildSerializationTool.serialize_mask_list(mask)

    def get_upload_parameters(self, gradient_mask, model_mask):
        upload_parameters = []
        for name, params in self._model.backbone.named_parameters():
            if "layer" in name:
                upload_parameters.append(gradient_mask[params] * params)
            else:
                upload_parameters.append(model_mask[name] * params)
        #         upload_parameters.append(params)
        return ChildSerializationTool.serialize_params_list(upload_parameters)

    @property
    def uplink_param_package(self):
        return self.param_list

    @property
    def uplink_mask_package(self):
        return self.masks_list


class FedChildClientManager(BaseClientManager, ABC):
    def __init__(self, network, trainer):
        super().__init__(network, trainer)

    def synchronize(self):
        """Synchronize with server"""
        self.logger.info("Uploading Parameters to server.")
        self._network.send(
            content=self._trainer.uplink_param_package,
            message_code=MessageCode.ParameterUpdate,
            dst=0
        )

        self.logger.info("Uploading Masks to server.")
        self._network.send(
            content=self._trainer.uplink_mask_package,
            message_code=MessageCode.ParameterUpdate,
            dst=0
        )
