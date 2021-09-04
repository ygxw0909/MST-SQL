import os, sys

sys.path.append("..")
import json
import pickle
from utils import read_conf
from modeling.torch_model import HydraTorch
from wikisql_lib.query import Query
from featurizer import HydraFeaturizer, SQLDataset
from wikisql_lib.dbengine import DBEngine


def print_metric(label_file, pred_file, db_file):
    sp = [(json.loads(ls)["sql"], json.loads(lp)["query"]) for ls, lp in zip(open(label_file), open(pred_file))]

    sel_acc = sum(p["sel"] == s["sel"] for s, p in sp) / len(sp)
    agg_acc = sum(p["agg"] == s["agg"] for s, p in sp) / len(sp)
    wcn_acc = sum(len(p["conds"]) == len(s["conds"]) for s, p in sp) / len(sp)

    def wcc_match(a, b):
        a = sorted(a, key=lambda k: k[0])
        b = sorted(b, key=lambda k: k[0])
        return [c[0] for c in a] == [c[0] for c in b]

    def wco_match(a, b):
        a = sorted(a, key=lambda k: k[0])
        b = sorted(b, key=lambda k: k[0])
        return [c[1] for c in a] == [c[1] for c in b]

    def wcv_match(a, b):
        a = sorted(a, key=lambda k: k[0])
        b = sorted(b, key=lambda k: k[0])
        return [str(c[2]).lower() for c in a] == [str(c[2]).lower() for c in b]

    wcc_acc = sum(wcc_match(p["conds"], s["conds"]) for s, p in sp) / len(sp)
    wco_acc = sum(wco_match(p["conds"], s["conds"]) for s, p in sp) / len(sp)
    wcv_acc = sum(wcv_match(p["conds"], s["conds"]) for s, p in sp) / len(sp)

    engine = DBEngine(db_file)
    exact_match = []
    grades = []
    with open(label_file) as fs, open(pred_file) as fp:
        for ls, lp in zip(fs, fp):
            eg = json.loads(ls)
            ep = json.loads(lp)
            qg = Query.from_dict(eg['sql'], ordered=False)
            gold = engine.execute_query(eg['table_id'], qg, lower=True)
            qp = Query.from_dict(ep['query'], ordered=False)
            try:
                pred = engine.execute_query(eg['table_id'], qp, lower=True)
            except Exception as e:
                pred = repr(e)
            correct = pred == gold
            match = qp == qg
            grades.append(correct)
            exact_match.append(match)

    res = 'ex_acc: {}\nlf_acc: {}\nsel_acc: {}\nagg_acc: {}\nwcn_acc: {}\nwcc_acc: {}\nwco_acc: {}\nwcv_acc: {}\n' \
        .format(sum(grades) / len(grades), sum(exact_match) / len(exact_match), sel_acc, agg_acc, wcn_acc, wcc_acc,
                wco_acc, wcv_acc)

    print(res)
    return res


def execute_one_test(dataset, shot, model_moment, epoch):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    model_path = "output/" + model_moment

    in_file = "data/wikisql/wiki{}_content.jsonl".format(dataset)
    db_file = "data/wikisql/{}.db".format(dataset)
    label_file = "data/wikisql/{}.jsonl".format(dataset)
    out_path = "predictions/{}_{}_{}_{}".format(model_moment, epoch, dataset, shot)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, "out.jsonl")
    eg_out_file = os.path.join(out_path, "out_eg.jsonl")
    model_out_file = os.path.join(out_path, "model_out.pkl")
    test_result_file = os.path.join(out_path, "result.txt")

    engine = DBEngine(db_file)
    config = read_conf(os.path.join(model_path, "model.conf"))
    # config = read_conf("../conf/wikisql_content.conf")
    # config["DEBUG"] = 1
    featurizer = HydraFeaturizer(config)
    pred_data = SQLDataset(in_file, config, featurizer, False)
    print("num of samples: {0}".format(len(pred_data.input_features)))

    model = HydraTorch(config)
    model.load(model_path, epoch)

    if "DEBUG" in config:
        model_out_file = model_out_file + ".partial"
    model_outputs = model.dataset_inference(pred_data)
    pickle.dump(model_outputs, open(model_out_file, "wb"))
    # model_outputs = pickle.load(open(model_out_file, "rb"))

    print("===HydraNet===")
    pred_sqls = model.predict_SQL(pred_data, model_outputs=model_outputs)
    with open(out_file, "w") as g:
        for pred_sql in pred_sqls:
            # print(pred_sql)
            result = {"query": {}}
            result["query"]["agg"] = int(pred_sql[0])
            result["query"]["sel"] = int(pred_sql[1])
            result["query"]["conds"] = [(int(cond[0]), int(cond[1]), str(cond[2])) for cond in pred_sql[2]]
            g.write(json.dumps(result) + "\n")
    normal_res = print_metric(label_file, out_file, db_file)

    print("===HydraNet+EG===")
    pred_sqls = model.predict_SQL_with_EG(engine, pred_data, model_outputs=model_outputs)
    with open(eg_out_file, "w") as g:
        for pred_sql in pred_sqls:
            # print(pred_sql)
            result = {"query": {}}
            result["query"]["agg"] = int(pred_sql[0])
            result["query"]["sel"] = int(pred_sql[1])
            result["query"]["conds"] = [(int(cond[0]), int(cond[1]), str(cond[2])) for cond in pred_sql[2]]
            g.write(json.dumps(result) + "\n")
    eg_res = print_metric(label_file, eg_out_file, db_file)

    with open(test_result_file, "w") as g:
        g.write("normal results:\n" + normal_res + "eg results:\n" + eg_res)


if __name__ == "__main__":
    shots = ["orig"]
    splits = ["dev", "test"]
    models = [("20210824_155829", 6)]

    for moment, epoch in models:
        for split in splits:
            for shot in shots:
                execute_one_test(split, shot, moment, epoch)

