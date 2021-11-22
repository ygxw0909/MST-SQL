import torch
import os
import numpy as np
import transformers
import utils
from modeling.base_model import BaseModel
from torch import nn
import torch.utils.data as torch_data

class HydraTorch(BaseModel):
    def __init__(self, config):
        self.config = config
        self.model = HydraNet(config)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer, self.scheduler = None, None
        if config["use_meta_semi"] == "True":
            self.meta_optimizer, self.meta_scheduler = None, None
        if config["use_col_enhance"] == "True":
            self.col_enhance_optimizer, self.col_enhance_scheduler = None, None

    def train_on_batch(self, batch):
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": float(self.config["decay"]),
                },
                {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0},
            ]
            self.optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=float(self.config["learning_rate"]))
            self.scheduler = transformers.get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(self.config["num_warmup_steps"]),
                num_training_steps=int(self.config["num_train_steps"]))
            self.optimizer.zero_grad()

        self.model.train()
        for k, v in batch.items():
            batch[k] = v.to(self.device)
        batch_loss = torch.mean(self.model(**batch)["loss"])
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return batch_loss.cpu().detach().numpy()

    def train_on_meta_semi_task(self, task):
        if self.meta_optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": float(self.config["decay"]),
                },
                {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0},
            ]
            self.meta_optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=float(self.config["meta_learning_rate"]))
            self.meta_scheduler = transformers.get_cosine_schedule_with_warmup(
                self.meta_optimizer,
                num_warmup_steps=int(self.config["num_warmup_steps"]),
                num_training_steps=int(self.config["num_train_steps"]))
            self.meta_optimizer.zero_grad()
        self.model.train()

        semi_data = task["semi"]
        train_data = task["train"]

        train_data_loader = torch_data.DataLoader(train_data, batch_size=int(self.config["batch_size"]), shuffle=True, pin_memory=False)
        semi_data_loader = torch_data.DataLoader(semi_data, batch_size=int(self.config["batch_size"]), shuffle=True, pin_memory=True)
        self.save(self.config["model_path"], "temp")
        loss_before, loss_after = 0.0, 0.0
        for train_batch in train_data_loader:
            for k, v in train_batch.items():
                train_batch[k] = v.to(self.device)
            batch_loss = torch.mean(self.model(**train_batch)["loss"])
            loss_before += float(batch_loss)
            torch.cuda.empty_cache()

        for semi_batch in semi_data_loader:
            for k, v in semi_batch.items():
                semi_batch[k] = v.to(self.device)
            batch_loss = torch.mean(self.model(**semi_batch)["loss"])
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.meta_optimizer.step()
            self.meta_scheduler.step()
            self.meta_optimizer.zero_grad()
            torch.cuda.empty_cache()

        for train_batch in train_data_loader:
            for k, v in train_batch.items():
                train_batch[k] = v.to(self.device)
            batch_loss = torch.mean(self.model(**train_batch)["loss"])
            loss_after += float(batch_loss)
            torch.cuda.empty_cache()

        if loss_before > loss_after:
            self.load(self.config["model_path"], "temp")
            return loss_before, False
        else:
            return loss_after, True

    def train_on_meta_semi_tasks(self, tasks):
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": float(self.config["decay"]),
                },
                {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0},
            ]
            self.optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=float(self.config["meta_learning_rate"]))
            self.scheduler = transformers.get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(self.config["num_warmup_steps"]),
                num_training_steps=int(self.config["num_train_steps"]))
            self.optimizer.zero_grad()
        self.model.train()

        self.save(self.config["model_path"], "temp")

        semi_loss_list = [0.0] * len(tasks)
        flag_list = [0] * len(tasks)

        for i, task in enumerate(tasks):

            semi_data = task["semi"]
            train_data = task["train"]

            train_data_loader = torch_data.DataLoader(train_data, batch_size=int(self.config["batch_size"]), shuffle=True, pin_memory=False)
            semi_data_loader = torch_data.DataLoader(semi_data, batch_size=int(self.config["batch_size"]), shuffle=True, pin_memory=True)
            loss_before, loss_after = 0.0, 0.0

            for train_batch in train_data_loader:
                for k, v in train_batch.items():
                    train_batch[k] = v.to(self.device)
                batch_loss = torch.mean(self.model(**train_batch)["loss"])
                loss_before += float(batch_loss)
                torch.cuda.empty_cache()

            for semi_batch in semi_data_loader:
                for k, v in semi_batch.items():
                    semi_batch[k] = v.to(self.device)
                batch_loss = torch.mean(self.model(**semi_batch)["loss"])
                semi_loss_list[i] += batch_loss
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()

            for train_batch in train_data_loader:
                for k, v in train_batch.items():
                    train_batch[k] = v.to(self.device)
                batch_loss = torch.mean(self.model(**train_batch)["loss"])
                loss_after += float(batch_loss)
                torch.cuda.empty_cache()

            if loss_before >= loss_after:
                flag_list[i] = 1
            self.load(self.config["model_path"], "temp")

        total_loss = 0.0
        for i, loss in enumerate(semi_loss_list):
            if flag_list[i] == 1:
                total_loss += semi_loss_list[i]

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()

        return total_loss, flag_list

    def train_on_column_force(self, task):
        self.model.train()
        if self.col_enhance_optimizer is None:
            no_decay = ["bias", "LayerNorm.weight", "layer.0", "layer.1", "layer.2", "layer.3", "layer.4", "layer.5", "layer.6", "layer.7", "layer.8", "layer.9",
                        "layer.10", "layer.11", "layer.12", "layer.13", "layer.14", "layer.15", "layer.16", "layer.17", "layer.18", "layer.19", "layer.20", "layer.21"]
            # no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": float(self.config["decay"]),
                },
                {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0},
            ]
            self.col_enhance_optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=float(self.config["meta_learning_rate"]))
            self.col_enhance_scheduler = transformers.get_cosine_schedule_with_warmup(
                self.col_enhance_optimizer,
                num_warmup_steps=int(self.config["num_warmup_steps"]),
                num_training_steps=int(self.config["num_train_steps"]))
            self.col_enhance_optimizer.zero_grad()

        # self.save(self.config["model_path"], "temp")
        s_loss, q_loss = 0.0, 0.0
        support_set = task["support"]
        query_set = task["query"]

        # self.save(model_path, "temp")

        support_set_loader = torch_data.DataLoader(support_set, batch_size=int(self.config["batch_size"]), shuffle=True, pin_memory=True)
        query_set_loader = torch_data.DataLoader(query_set, batch_size=int(self.config["batch_size"]), shuffle=True, pin_memory=True)

        for support_batch in support_set_loader:
            for k, v in support_batch.items():
                support_batch[k] = v.to(self.device)

            loss = torch.mean(self.model(**support_batch)["loss"])
            s_loss += loss
            loss.backward(retain_graph=True)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.col_enhance_optimizer.step()
        self.col_enhance_scheduler.step()
        self.col_enhance_optimizer.zero_grad()

        for query_batch in query_set_loader:
            for k, v in query_batch.items():
                query_batch[k] = v.to(self.device)

            q_loss += torch.mean(self.model(**query_batch)["loss"])

        meta_loss = 0.5 * s_loss + 0.5 * q_loss
        # meta_loss = (1 - beta) * q_loss
        meta_loss.backward()

        # self.load(model_path, "temp")

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.col_enhance_optimizer.step()
        self.col_enhance_scheduler.step()
        self.col_enhance_optimizer.zero_grad()
        torch.cuda.empty_cache()
        return float(meta_loss)


    def model_inference(self, model_inputs):
        self.model.eval()
        model_outputs = {}
        batch_size = 512
        for start_idx in range(0, model_inputs["input_ids"].shape[0], batch_size):
            input_tensor = {k: torch.from_numpy(model_inputs[k][start_idx:start_idx+batch_size]).to(self.device) for k in ["input_ids", "input_mask", "segment_ids"]}
            with torch.no_grad():
                model_output = self.model(**input_tensor)
            for k, out_tensor in model_output.items():
                if out_tensor is None:
                    continue
                if k not in model_outputs:
                    model_outputs[k] = []
                model_outputs[k].append(out_tensor.cpu().detach().numpy())

        for k in model_outputs:
            model_outputs[k] = np.concatenate(model_outputs[k], 0)

        return model_outputs

    def save(self, model_path, epoch):
        if "SAVE" in self.config and "DEBUG" not in self.config:
            save_path = os.path.join(model_path, "model_{0}.pt".format(epoch))
            if torch.cuda.device_count() > 1:
                torch.save(self.model.module.state_dict(), save_path)
            else:
                torch.save(self.model.state_dict(), save_path)
            print("Model saved in path: %s" % save_path)

    def load(self, model_path, epoch):
        pt_path = os.path.join(model_path, "model_{0}.pt".format(epoch))
        loaded_dict = torch.load(pt_path, map_location=torch.device(self.device))
        if torch.cuda.device_count() > 1:
            self.model.module.load_state_dict(loaded_dict)
        else:
            self.model.load_state_dict(loaded_dict)
        print("PyTorch model loaded from {0}".format(pt_path))

class HydraNet(nn.Module):
    def __init__(self, config):
        super(HydraNet, self).__init__()
        self.config = config
        self.base_model = utils.create_base_model(config)

        # #=====Hack for RoBERTa model====
        # self.base_model.config.type_vocab_size = 2
        # single_emb = self.base_model.embeddings.token_type_embeddings
        # self.base_model.embeddings.token_type_embeddings = torch.nn.Embedding(2, single_emb.embedding_dim)
        # self.base_model.embeddings.token_type_embeddings.weight = torch.nn.Parameter(single_emb.weight.repeat([2, 1]), requires_grad=True)
        # #====================================

        drop_rate = float(config["drop_rate"]) if "drop_rate" in config else 0.0
        self.dropout = nn.Dropout(drop_rate)

        bert_hid_size = self.base_model.config.hidden_size
        self.column_func = nn.Linear(bert_hid_size, 3)
        self.agg = nn.Linear(bert_hid_size, int(config["agg_num"]))
        self.op = nn.Linear(bert_hid_size, int(config["op_num"]))
        self.where_num = nn.Linear(bert_hid_size, int(config["where_column_num"]) + 1)
        self.start_end = nn.Linear(bert_hid_size, 2)

    def forward(self, input_ids, input_mask, segment_ids, agg=None, select=None, where=None, where_num=None, op=None, value_start=None, value_end=None, confidence=None, col_relevance=None):
        # print("[inner] input_ids size:", input_ids.size())
        if self.config["base_class"] in ["roberta", "grappa", "roberta_cn"]:
            bert_output, pooled_output = self.base_model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=None,
                return_dict=False)
        elif self.config["base_class"] == "tapas":
            bert_output, pooled_output = self.base_model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=utils.convert_tapas_segment_ids(
                    segment_ids).to(torch.device("cuda")),
                return_dict=False
            )
        else:
            bert_output, pooled_output = self.base_model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                return_dict=False)

        bert_output = self.dropout(bert_output)
        pooled_output = self.dropout(pooled_output)

        column_func_logit = self.column_func(pooled_output)
        agg_logit = self.agg(pooled_output)
        op_logit = self.op(pooled_output)
        where_num_logit = self.where_num(pooled_output)
        start_end_logit = self.start_end(bert_output)
        value_span_mask = input_mask.to(dtype=bert_output.dtype)
        # value_span_mask[:, 0] = 1
        start_logit = start_end_logit[:, :, 0] * value_span_mask - 1000000.0 * (1 - value_span_mask)
        end_logit = start_end_logit[:, :, 1] * value_span_mask - 1000000.0 * (1 - value_span_mask)

        loss = None
        if select is not None:
            bceloss = nn.BCEWithLogitsLoss(reduction="none")
            cross_entropy = nn.CrossEntropyLoss(reduction="none")

            if self.config["use_col_enhance"] == "True" and self.config["state"] == "train_col":
                loss = bceloss(column_func_logit[:, 0], select.float())
                loss += bceloss(column_func_logit[:, 1], where.float())
                # loss += bceloss(column_func_logit[:, 2], (1 - select.float()) * (1 - where.float()))
                # print("train_col")
            else:
                loss = cross_entropy(agg_logit, agg) * select.float()
                loss += bceloss(column_func_logit[:, 0], select.float())
                loss += bceloss(column_func_logit[:, 1], where.float())
                loss += bceloss(column_func_logit[:, 2], (1-select.float()) * (1-where.float()))
                loss += cross_entropy(where_num_logit, where_num)
                loss += cross_entropy(op_logit, op) * where.float()
                loss += cross_entropy(start_logit, value_start)
                loss += cross_entropy(end_logit, value_end)

            if self.config["use_semi"] == "True":
                loss *= confidence.float()

            if self.config["use_col_enhance"] == "True" and self.config["state"] == "train_col":
                loss *= col_relevance.float()

        # return loss, column_func_logit, agg_logit, op_logit, where_num_logit, start_logit, end_logit
        log_sigmoid = nn.LogSigmoid()

        return {"column_func": log_sigmoid(column_func_logit),
                "agg": agg_logit.log_softmax(1),
                "op": op_logit.log_softmax(1),
                "where_num": where_num_logit.log_softmax(1),
                "value_start": start_logit.log_softmax(1),
                "value_end": end_logit.log_softmax(1),
                "loss": loss}