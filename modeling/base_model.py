import numpy as np
import time
from typing import List
from featurizer import SQLDataset

class BaseModel(object):
    """Define common interfaces for HydraNet models"""
    def train_on_meta_semi_task(self, task):
        raise NotImplementedError()

    def train_on_meta_semi_tasks(self, tasks):
        raise NotImplementedError()

    def train_on_column_force(self, task):
        raise NotImplementedError()

    def train_on_batch(self, batch):
        raise NotImplementedError()

    def save(self, model_path, epoch):
        raise NotImplementedError()

    def load(self, model_path, epoch):
        raise NotImplementedError()

    def model_inference(self, model_inputs):
        """model prediction on processed features"""
        raise NotImplementedError()

    def dataset_inference(self, dataset: SQLDataset):
        print("model prediction start")
        start_time = time.time()
        model_outputs = self.model_inference(dataset.model_inputs)

        final_outputs = []
        for pos in dataset.pos:
            final_output = {}
            for k in model_outputs:
                final_output[k] = model_outputs[k][pos[0]:pos[1], :]
            final_outputs.append(final_output)
        print("model prediction end, time elapse: {0}".format(time.time() - start_time))
        assert len(dataset.input_features) == len(final_outputs)

        return final_outputs

    def predict_SQL(self, dataset: SQLDataset, model_outputs=None):
        if model_outputs is None:
            model_outputs = self.dataset_inference(dataset)
        sqls = []
        for input_feature, model_output in zip(dataset.input_features, model_outputs):
            agg, select, where, conditions , _= self.parse_output(input_feature, model_output, [])

            conditions_with_value_texts = []
            for wc in where:
                _, op, vs, ve = conditions[wc]
                word_start, word_end = input_feature.subword_to_word[wc][vs], input_feature.subword_to_word[wc][ve]
                char_start = input_feature.word_to_char_start[word_start]
                char_end = len(input_feature.question)
                if word_end + 1 < len(input_feature.word_to_char_start):
                    char_end = input_feature.word_to_char_start[word_end + 1]
                value_span_text = input_feature.question[char_start:char_end].rstrip()
                conditions_with_value_texts.append((wc, op, value_span_text))

            sqls.append((agg, select, conditions_with_value_texts))

        return sqls

    def predict_SQL_with_EG(self, engine, dataset: SQLDataset, beam_size=5, model_outputs=None):
        if model_outputs is None:
            model_outputs = self.dataset_inference(dataset)
        sqls = []
        for input_feature, model_output in zip(dataset.input_features, model_outputs):
            agg, select, where_num, conditions = self.beam_parse_output(input_feature, model_output, beam_size)
            query = {"agg": agg, "sel": select, "conds": []}
            wcs = set()
            conditions_with_value_texts = []
            for condition in conditions:
                if len(wcs) >= where_num:
                    break
                _, wc, op, vs, ve = condition
                if wc in wcs:
                    continue

                word_start, word_end = input_feature.subword_to_word[wc][vs], input_feature.subword_to_word[wc][ve]
                char_start = input_feature.word_to_char_start[word_start]
                char_end = len(input_feature.question)
                if word_end + 1 < len(input_feature.word_to_char_start):
                    char_end = input_feature.word_to_char_start[word_end + 1]
                value_span_text = input_feature.question[char_start:char_end].rstrip()

                query["conds"] = [[int(wc), int(op), value_span_text]]
                result, sql = engine.execute_dict_query(input_feature.table_id, query)
                if not result or 'ERROR: ' in result:
                    continue

                conditions_with_value_texts.append((wc, op, value_span_text))
                wcs.add(wc)

            sqls.append((agg, select, conditions_with_value_texts))

        return sqls

    def predict_for_unlabeled_data(self, semi_data: SQLDataset):
        column_count_dict = {}
        table_list = []
        col_num = 0
        for input_feature in semi_data.input_features:
            if input_feature.table_id not in table_list:
                table_list.append(input_feature.table_id)
                for col in input_feature.columns:
                    if col.lower() not in column_count_dict:
                        column_count_dict[col.lower()] = 1
                    else:
                        column_count_dict[col.lower()] += 1
                    col_num += 1
            else:
                continue
        col_num_mean = col_num / len(column_count_dict.keys())


        key = ["agg", "select", "where_num", "where", "op", "value_start", "value_end", "confidence", "col_relevance"]
        for k in key:
            semi_data.model_inputs[k] = []

        model_outputs = self.dataset_inference(semi_data)
        for i, (input_feature, model_output) in enumerate(zip(semi_data.input_features, model_outputs)):
            # print("model_output", model_output)
            agg, select, where, conditions, confidence = self.parse_output(input_feature, model_output, [])
            column_num = len(input_feature.columns)
            input_feature.select = [0] * column_num
            input_feature.select[select] = 1

            input_feature.agg = [0] * column_num
            input_feature.agg[select] = agg

            input_feature.where_num = [len(conditions)] * column_num

            input_feature.where = [0] * column_num
            input_feature.op = [0] * column_num
            input_feature.value_start = [0] * column_num
            input_feature.value_end = [0] * column_num

            for col in conditions.keys():
                _, op, value_start, value_end = conditions[col]
                input_feature.where[col] = 1
                input_feature.op[col] = op
                input_feature.value_start[col] = value_start
                input_feature.value_end[col] = value_end

            # print("keys:", semi_data.model_inputs.keys())

            semi_data.model_inputs["select"].extend(input_feature.select)
            semi_data.model_inputs["agg"].extend(input_feature.agg)
            semi_data.model_inputs["where_num"].extend(input_feature.where_num)
            semi_data.model_inputs["where"].extend(input_feature.where)
            semi_data.model_inputs["op"].extend(input_feature.op)
            semi_data.model_inputs["value_start"].extend(input_feature.value_start)
            semi_data.model_inputs["value_end"].extend(input_feature.value_end)
            semi_data.model_inputs["confidence"].extend([confidence] * column_num)
            for col in input_feature.columns:
                if col.lower() not in column_count_dict:
                    print("column name error, semi prediction")
                    semi_data.model_inputs["col_relevance"].append(0)
                else:
                    col_rel = min(0.2, column_count_dict[col.lower()] / col_num_mean, 2.0)
                    semi_data.model_inputs["col_relevance"].append(col_rel)

        for k in key:
            if k not in ["confidence", "col_relevance"]:
                semi_data.model_inputs[k] = np.array(semi_data.model_inputs[k], dtype=np.int64)
            else:
                semi_data.model_inputs[k] = np.array(semi_data.model_inputs[k], dtype=np.float)

    def _get_where_num(self, output):
        # wn = np.argmax(output["where_num"], -1)
        # max_num = 0
        # max_cnt = np.sum(wn == 0)
        # for num in range(1, 5):
        #     cur_cnt = np.sum(wn==num)
        #     if cur_cnt > max_cnt:
        #         max_cnt = cur_cnt
        #         max_num = num
        # def sigmoid(x):
        #     return 1/(1 + np.exp(-x))
        relevant_prob = 1 - np.exp(output["column_func"][:, 2])
        where_num_scores = np.average(output["where_num"], axis=0, weights=relevant_prob)
        where_num = int(np.argmax(where_num_scores))

        return where_num

    def parse_output(self, input_feature, model_output, where_label = []):
        def get_span(i):
            offset = 0
            segment_ids = np.array(input_feature.segment_ids[i])
            for j in range(len(segment_ids)):
                if segment_ids[j] == 1:
                    offset = j
                    break

            value_start, value_end = model_output["value_start"][i, segment_ids == 1], model_output["value_end"][i, segment_ids == 1]
            l = len(value_start)
            sum_mat = value_start.reshape((l, 1)) + value_end.reshape((1, l))
            span = (0, 0)
            for cur_span, _ in sorted(np.ndenumerate(sum_mat), key=lambda x:x[1], reverse=True):
                if cur_span[1] < cur_span[0] or cur_span[0] == l - 1 or cur_span[1] == l - 1:
                    continue
                span = cur_span
                break

            return (span[0]+offset, span[1]+offset)

        select_id_prob = sorted(enumerate(model_output["column_func"][:, 0]), key=lambda x:x[1], reverse=True)
        select = select_id_prob[0][0]
        agg = np.argmax(model_output["agg"][select, :])

        where_id_prob = sorted(enumerate(model_output["column_func"][:, 1]), key=lambda x:x[1], reverse=True)
        where_num = self._get_where_num(model_output)
        where = [i for i, _ in where_id_prob[:where_num]]
        conditions = {}
        for idx in set(where + where_label):
            span = get_span(idx)
            op = np.argmax(model_output["op"][idx, :])
            conditions[idx] = (idx, op, span[0], span[1])
        confidence = (max(select_id_prob[0][1], 0.1) * max(where_id_prob[0][1], 0.1) * max(model_output["agg"][select][agg], 0.1))
        # confidence = (max(select_id_prob[0][1], 0.1) * max(where_id_prob[0][1], 0.1))

        return agg, select, where, conditions, pow(confidence, 0.4)

    def beam_parse_output(self, input_feature, model_output, beam_size=5):
        def get_span(i):
            offset = 0
            segment_ids = np.array(input_feature.segment_ids[i])
            for j in range(len(segment_ids)):
                if segment_ids[j] == 1:
                    offset = j
                    break

            value_start, value_end = model_output["value_start"][i, segment_ids == 1], model_output["value_end"][i, segment_ids == 1]
            l = len(value_start)
            sum_mat = value_start.reshape((l, 1)) + value_end.reshape((1, l))
            spans = []
            for cur_span, sum_logp in sorted(np.ndenumerate(sum_mat), key=lambda x:x[1], reverse=True):
                if cur_span[1] < cur_span[0] or cur_span[0] == l - 1 or cur_span[1] == l - 1:
                    continue
                spans.append((cur_span[0]+offset, cur_span[1]+offset, sum_logp))
                if len(spans) >= beam_size:
                    break

            return spans

        select_id_prob = sorted(enumerate(model_output["column_func"][:, 0]), key=lambda x:x[1], reverse=True)
        select = select_id_prob[0][0]
        agg = np.argmax(model_output["agg"][select, :])

        where_id_prob = sorted(enumerate(model_output["column_func"][:, 1]), key=lambda x:x[1], reverse=True)
        where_num = self._get_where_num(model_output)
        conditions = []
        for idx, wlogp in where_id_prob[:beam_size]:
            op = np.argmax(model_output["op"][idx, :])
            for span in get_span(idx):
                conditions.append((wlogp+span[2], idx, op, span[0], span[1]))
        conditions.sort(key=lambda x:x[0], reverse=True)
        return agg, select, where_num, conditions
