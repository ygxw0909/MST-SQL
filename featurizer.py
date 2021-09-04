import copy
import numpy as np
import json
import os
import utils
import torch.utils.data as torch_data
from collections import defaultdict
import random

stats = defaultdict(int)

class SQLExample(object):
    def __init__(self,
                 qid,
                 question,
                 table_id,
                 column_meta,
                 agg=None,
                 select=None,
                 conditions=None,
                 tokens=None,
                 char_to_word=None,
                 word_to_char_start=None,
                 value_start_end=None,
                 valid=True,
                 content=None,
                 is_cn=False):
        self.qid = qid
        self.question = question
        self.table_id = table_id
        self.column_meta = column_meta
        self.agg = agg
        self.select = select
        self.conditions = conditions
        self.valid = valid
        self.content = content
        if tokens is None:
            if not is_cn:
                self.tokens, self.char_to_word, self.word_to_char_start = utils.basic_tokenize(question)
            else:
                self.tokens, self.char_to_word, self.word_to_char_start = utils.basic_tokenize_cn(question)
            self.value_start_end = {}
            if conditions is not None and len(conditions) > 0:
                cur_start = None
                for cond in conditions:
                    value = cond[-1]
                    if not is_cn:
                        value_tokens, _, _ = utils.basic_tokenize(value)
                    else:
                        value_tokens, _, _ = utils.basic_tokenize_cn(value)
                    val_len = len(value_tokens)
                    for i in range(len(self.tokens)):
                        if " ".join(self.tokens[i:i+val_len]).lower() != " ".join(value_tokens).lower():
                            continue
                        s = self.word_to_char_start[i]
                        e = len(question) if i + val_len >= len(self.word_to_char_start) else self.word_to_char_start[i + val_len]
                        recovered_answer_text = question[s:e].strip()
                        if value.lower() == recovered_answer_text.lower():
                            cur_start = i
                            break

                    if cur_start is None:
                        self.valid = False
                        print([value, value_tokens, question, self.tokens])
                        # for c in question:
                        #     print((c, ord(c), unicodedata.category(c)))
                        # raise Exception()
                    else:
                        self.value_start_end[value] = (cur_start, cur_start + val_len)
        else:
            self.tokens, self.char_to_word, self.word_to_char_start, self.value_start_end = tokens, char_to_word, word_to_char_start, value_start_end

    @staticmethod
    def load_from_json(s):
        d = json.loads(s)
        keys = ["qid", "question", "table_id", "column_meta", "agg", "select", "conditions", "tokens", "char_to_word", "word_to_char_start", "value_start_end", "valid", "content"]

        return SQLExample(*[d[k] for k in keys])

    def dump_to_json(self):
        d = {}
        d["qid"] = self.qid
        d["question"] = self.question
        d["table_id"] = self.table_id
        d["column_meta"] = self.column_meta
        d["agg"] = self.agg
        d["select"] = self.select
        d["conditions"] = self.conditions
        d["tokens"] = self.tokens
        d["char_to_word"] = self.char_to_word
        d["word_to_char_start"] = self.word_to_char_start
        d["value_start_end"] = self.value_start_end
        d["valid"] = self.valid
        d["content"] = self.content

        return json.dumps(d, ensure_ascii=False)

    def output_SQ(self, return_str=True):
        agg_ops = ['NA', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        cond_ops = ['=', '>', '<', 'OP']

        agg_text = agg_ops[self.agg]
        select_text = self.column_meta[self.select][0]
        cond_texts = []
        for wc, op, value_text in self.conditions:
            column_text = self.column_meta[wc][0]
            op_text = cond_ops[op]
            cond_texts.append(column_text + op_text + value_text)

        if return_str:
            sq = agg_text + ", " + select_text + ", " + " AND ".join(cond_texts)
        else:
            sq = (agg_text, select_text, set(cond_texts))
        return sq


class InputFeature(object):
    def __init__(self,
                 question,
                 table_id,
                 tokens,
                 word_to_char_start,
                 word_to_subword,
                 subword_to_word,
                 input_ids,
                 input_mask,
                 segment_ids):
        self.question = question
        self.table_id = table_id
        self.tokens = tokens
        self.word_to_char_start = word_to_char_start
        self.word_to_subword = word_to_subword
        self.subword_to_word = subword_to_word
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

        self.columns = None
        self.agg = None
        self.select = None
        self.where_num = None
        self.where = None
        self.op = None
        self.value_start = None
        self.value_end = None

    def output_SQ(self, agg = None, sel = None, conditions = None, return_str=True):
        agg_ops = ['NA', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        cond_ops = ['=', '>', '<', 'OP']

        if agg is None and sel is None and conditions is None:
            sel = np.argmax(self.select)
            agg = self.agg[sel]
            conditions = []
            for i in range(len(self.where)):
                if self.where[i] == 0:
                    continue
                conditions.append((i, self.op[i], self.value_start[i], self.value_end[i]))

        agg_text = agg_ops[agg]
        select_text = self.columns[sel]
        cond_texts = []
        for wc, op, vs, ve in conditions:
            column_text = self.columns[wc]
            op_text = cond_ops[op]
            word_start, word_end = self.subword_to_word[wc][vs], self.subword_to_word[wc][ve]
            char_start = self.word_to_char_start[word_start]
            char_end = len(self.question) if word_end + 1 >= len(self.word_to_char_start) else self.word_to_char_start[word_end + 1]
            value_span_text = self.question[char_start:char_end]
            cond_texts.append(column_text + op_text + value_span_text.rstrip())

        if return_str:
            sq = agg_text + ", " + select_text + ", " + " AND ".join(cond_texts)
        else:
            sq = (agg_text, select_text, set(cond_texts))

        return sq

class HydraFeaturizer(object):
    def __init__(self, config):
        self.config = config
        self.tokenizer = utils.create_tokenizer(config)
        self.colType2token = {
            "string": "[unused1]",
            "real": "[unused2]"}

    def get_input_feature(self, example: SQLExample, config):
        max_total_length = int(config["max_total_length"])

        input_feature = InputFeature(
            example.question,
            example.table_id,
            [],
            example.word_to_char_start,
            [],
            [],
            [],
            [],
            []
        )

        use_content = "use_content" in config.keys() and config["use_content"] == "True"
        input_feature.columns = [c[0] for c in example.column_meta]
        contents = example.content if use_content else None

        for i, (column, col_type, _) in enumerate(example.column_meta):
            # get query tokens
            tokens = []
            word_to_subword = []
            subword_to_word = []
            content = contents[i] if use_content else None

            for i, query_token in enumerate(example.tokens):
                if self.config["base_class"] == ["roberta", "grappa"]:
                    sub_tokens = self.tokenizer.tokenize(query_token, add_prefix_space=True)
                else:
                    sub_tokens = self.tokenizer.tokenize(query_token)
                cur_pos = len(tokens)
                if len(sub_tokens) > 0:
                    word_to_subword += [(cur_pos, cur_pos + len(sub_tokens))]
                    tokens.extend(sub_tokens)
                    subword_to_word.extend([i] * len(sub_tokens))

            column_input = col_type + " " + column + " " + " ".join(content) if use_content else col_type + " " + column

            if self.config["base_class"] == "grappa":
                tokenize_result = self.tokenizer.encode_plus(
                    column_input,
                    tokens,
                    padding="max_length",
                    max_length=max_total_length,
                    truncation=True,
                )
            elif self.config["base_class"] == "roberta":
                tokenize_result = self.tokenizer.encode_plus(
                    column_input,
                    tokens,
                    padding="max_length",
                    max_length=max_total_length,
                    truncation=True,
                    add_prefix_space=True
                )
            else:
                tokenize_result = self.tokenizer.encode_plus(
                    column_input,
                    tokens,
                    padding="max_length",
                    max_length=max_total_length,
                    truncation=True,
                )

            input_ids = tokenize_result["input_ids"]
            input_mask = tokenize_result["attention_mask"]

            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            column_token_length = 0
            for i, token_id in enumerate(input_ids):
                if token_id == self.tokenizer.sep_token_id:
                    column_token_length = i + 2
                    break
            segment_ids = [0] * max_total_length
            for i in range(column_token_length, max_total_length):
                if input_mask[i] == 0:
                    break
                segment_ids[i] = 1

            subword_to_word = [0] * column_token_length + subword_to_word
            word_to_subword = [(pos[0]+column_token_length, pos[1]+column_token_length) for pos in word_to_subword]

            assert len(input_ids) == max_total_length
            assert len(input_mask) == max_total_length
            assert len(segment_ids) == max_total_length

            input_feature.tokens.append(tokens)
            input_feature.word_to_subword.append(word_to_subword)
            input_feature.subword_to_word.append(subword_to_word)
            input_feature.input_ids.append(input_ids)
            input_feature.input_mask.append(input_mask)
            input_feature.segment_ids.append(segment_ids)

        return input_feature

    def fill_label_feature(self, example: SQLExample, input_feature: InputFeature, config):
        max_total_length = int(config["max_total_length"])

        col_num = len(input_feature.columns)

        input_feature.agg = [0] * col_num
        input_feature.agg[example.select] = example.agg
        input_feature.where_num = [len(example.conditions)] * col_num

        input_feature.select = [0] * col_num
        input_feature.select[example.select] = 1

        input_feature.where = [0] * col_num
        input_feature.op = [0] * col_num
        input_feature.value_start = [0] * col_num
        input_feature.value_end = [0] * col_num

        for colidx, op, _ in example.conditions:
            input_feature.where[colidx] = 1
            input_feature.op[colidx] = op
        for colidx, column_meta in enumerate(example.column_meta):
            if column_meta[-1] == None:
                continue
            se = example.value_start_end[column_meta[-1]]
            try:
                s = input_feature.word_to_subword[colidx][se[0]][0]
                input_feature.value_start[colidx] = s
                e = input_feature.word_to_subword[colidx][se[1]-1][1]-1
                input_feature.value_end[colidx] = e

                assert s < max_total_length and input_feature.input_mask[colidx][s] == 1
                assert e < max_total_length and input_feature.input_mask[colidx][e] == 1

            except:
                print("value span is out of range")
                return False

        # feature_sq = input_feature.output_SQ(return_str=False)
        # example_sq = example.output_SQ(return_str=False)
        # if feature_sq != example_sq:
        #     print(example.qid, feature_sq, example_sq)
        return True

    def load_data(self, data_paths, config, include_label=False):
        column_count_dict = {}
        col_num = 0
        table_list = []
        for data_path in data_paths.split("|"):
            for line in open(data_path, encoding="utf8"):
                t = json.loads(line.strip())
                if t["table_id"] not in table_list:
                    table_list.append(t["table_id"])
                    for col in [c[0] for c in t["column_meta"]]:
                        if col.lower() not in column_count_dict:
                            column_count_dict[col.lower()] = 1
                        else:
                            column_count_dict[col.lower()] += 1
                        col_num += 1
                else:
                    continue
        col_num_mean = col_num / len(column_count_dict.keys())
        print("col_num:", col_num)
        print("col_num_mean:", col_num_mean)

        model_inputs = {k: [] for k in ["input_ids", "input_mask", "segment_ids"]}
        if include_label:
            for k in ["agg", "select", "where_num", "where", "op", "value_start", "value_end", "confidence", "col_relevance"]:
                model_inputs[k] = []

        pos = []
        input_features = []
        table_list = []
        for data_path in data_paths.split("|"):
            cnt = 0
            for line in open(data_path, encoding="utf8"):
                example = SQLExample.load_from_json(line)
                if not example.valid and include_label == True:
                    continue

                input_feature = self.get_input_feature(example, config)
                if include_label:
                    success = self.fill_label_feature(example, input_feature, config)
                    if not success:
                        continue

                # sq = input_feature.output_SQ()
                input_features.append(input_feature)

                cur_start = len(model_inputs["input_ids"])
                cur_sample_num = len(input_feature.input_ids)
                pos.append((cur_start, cur_start + cur_sample_num))
                table_list.extend([input_feature.table_id] * len(input_feature.input_ids))

                model_inputs["input_ids"].extend(input_feature.input_ids)
                model_inputs["input_mask"].extend(input_feature.input_mask)
                model_inputs["segment_ids"].extend(input_feature.segment_ids)
                if include_label:
                    model_inputs["agg"].extend(input_feature.agg)
                    model_inputs["select"].extend(input_feature.select)
                    model_inputs["where_num"].extend(input_feature.where_num)
                    model_inputs["where"].extend(input_feature.where)
                    model_inputs["op"].extend(input_feature.op)
                    model_inputs["value_start"].extend(input_feature.value_start)
                    model_inputs["value_end"].extend(input_feature.value_end)
                    model_inputs["confidence"].extend([1.0] * len(input_feature.select))
                    for col in input_feature.columns:
                        if col.lower() not in column_count_dict:
                            print("column name error")
                            model_inputs["col_relevance"].append(0)
                        else:
                            col_rel = max(0.2, min(column_count_dict[col.lower()] / col_num_mean, 2.0))
                            model_inputs["col_relevance"].append(col_rel)
                cnt += 1
                if cnt % 5000 == 0:
                    print(cnt)

                if "DEBUG" in config and cnt > 100:
                    break

        for k in model_inputs:
            if k not in ["confidence", "col_relevance"]:
                model_inputs[k] = np.array(model_inputs[k], dtype=np.int64)
            else:
                model_inputs[k] = np.array(model_inputs[k], dtype=np.float)

        return input_features, model_inputs, pos, table_list

class SQLDataset(torch_data.Dataset):
    def __init__(self, data_paths, config, featurizer, include_label=False):
        self.config = config
        self.featurizer = featurizer
        self.input_features, self.model_inputs, self.pos, self.table_list = self.featurizer.load_data(data_paths, config, include_label)

        print("{0} loaded. Data shapes:".format(data_paths))
        for k, v in self.model_inputs.items():
            print(k, v.shape)

    def __len__(self):
        return self.model_inputs["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.model_inputs.items()}

    def get_meta_semi_task(self, semi_data):
        task_step, task_batch_size, train_sample_num, semi_sample_num = int(self.config["task_step"]), int(self.config["task_batch_size"]), int(self.config["train_sample_num"]), int(self.config["semi_sample_num"])
        train_data_num = len(self.model_inputs["select"])
        semi_data_num = len(semi_data.model_inputs["select"])
        task_data = []
        for i in range(task_step):
            tasks = []
            for j in range(task_batch_size):
                task = {"train": [], "semi": []}
                sampling_train_idx = random.sample(range(train_data_num), train_sample_num)
                sampling_semi_idx = random.sample(range(semi_data_num), semi_sample_num)

                for train_idx in sampling_train_idx:
                    task["train"].append({k: v[train_idx] for k, v in self.model_inputs.items()})
                for semi_idx in sampling_semi_idx:
                    task["semi"].append({k: v[semi_idx] for k, v in semi_data.model_inputs.items()})
                tasks.append(task)
            task_data.append(tasks)
        return task_data

    def get_col_enhance_task(self, semi_data=None):
        task_step, n_way, k_shot, q_shot = int(self.config["task_step"]), int(self.config["n_way"]), int(self.config["k_shot"]), int(self.config["q_shot"])
        meta_data_dict = {}
        table_id_list = []
        if semi_data is None:
            # for k in self.model_inputs.keys():
            #     self.model_inputs[k] = self.model_inputs[k].tolist()
            for i, input_feature in enumerate(self.input_features):
                table_id = input_feature.table_id
                if table_id not in meta_data_dict:
                    meta_data_dict[table_id] = []
                for j in range(self.pos[i][0], self.pos[i][1]):
                    meta_data_dict[table_id].append({k: v[j] for k, v in self.model_inputs.items()})

        else:
            train_model_inputs = [{k: v[i] for k, v in self.model_inputs.items()} for i in range(len(self.table_list))]
            train_table_list = copy.deepcopy(self.table_list)
            semi_num = len(semi_data.model_inputs["select"])
            # semi_sampling_list = random.sample(range(0, semi_num), int(semi_num * float(self.config["sampling_rate"])))

            for i in range(semi_num):
                train_model_inputs.append({k: v[i] for k, v in semi_data.model_inputs.items()})
                train_table_list.append(semi_data.table_list[i])

            for i, model_input in enumerate(train_model_inputs):
                table_id = train_table_list[i]
                if table_id not in meta_data_dict:
                    meta_data_dict[table_id] = []
                meta_data_dict[table_id].append(model_input)

        meta_data_filter_dict = {}
        for k, v in meta_data_dict.items():
            if len(v) >= k_shot + q_shot:
                meta_data_filter_dict[k] = v
                table_id_list.append(k)

        task_list = []
        for i in range(task_step):
            sampling_table_id = random.sample(table_id_list, n_way)
            task = {"support": [], "query": []}
            for table_id in sampling_table_id:

                sample_list = random.sample(meta_data_filter_dict[table_id], k_shot + q_shot)
                task["support"].extend(sample_list[:k_shot])
                task["query"].extend(sample_list[k_shot:])
            task_list.append(task)

        return task_list

    def get_semi_data(self, semi_data):
        train_model_inputs = [{k: v[i] for k, v in self.model_inputs.items()} for i in range(len(self.model_inputs["select"]))]

        semi_num = len(semi_data.model_inputs["select"])
        semi_sampling_list = random.sample(range(0, semi_num), int(semi_num * float(self.config["sampling_rate"])))

        for i in semi_sampling_list:
            train_model_inputs.append({k: v[i] for k, v in semi_data.model_inputs.items()})

        return train_model_inputs
