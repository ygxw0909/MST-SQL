import argparse
import copy
import os
import sys
import shutil
import datetime
import utils
from modeling.model_factory import create_model
from featurizer import HydraFeaturizer, SQLDataset
from evaluator import HydraEvaluator
import torch.utils.data as torch_data
import random

parser = argparse.ArgumentParser(description='HydraNet training script')
parser.add_argument("--job", type=str, default="train", help="job can be train")
parser.add_argument("--conf", default="conf/wikisql.conf", help="conf file path")
parser.add_argument("--output_path", type=str, default="output", help="folder path for all outputs")
parser.add_argument("--model_path", help="trained model folder path (used in eval, predict and export mode)")
parser.add_argument("--epoch", help="epochs to restore (used in eval, predict and export mode)")
parser.add_argument("--gpu", type=str, default=None, help="gpu id")
parser.add_argument("--note", type=str)

args = parser.parse_args()

if args.job == "train":

    conf_path = os.path.abspath(args.conf)
    config = utils.read_conf(conf_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]
    # os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

    note = args.note if args.note else ""

    script_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    output_path = args.output_path
    model_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_path, model_name)

    config["model_path"] = model_path

    if "DEBUG" not in config:
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        shutil.copyfile(conf_path, os.path.join(model_path, "model.conf"))
        for pyfile in ["featurizer.py"]:
            shutil.copyfile(pyfile, os.path.join(model_path, pyfile))
        if config["model_type"] == "pytorch":
            shutil.copyfile("modeling/torch_model.py", os.path.join(model_path, "torch_model.py"))
        elif config["model_type"] == "tf":
            shutil.copyfile("modeling/tf_model.py", os.path.join(model_path, "tf_model.py"))
        else:
            raise Exception("model_type is not supported")

    featurizer = HydraFeaturizer(config)
    train_data = SQLDataset(config["train_data_path"], config, featurizer, True)
    print("train_data_num: ", len(train_data))
    # train_data_loader = torch_data.DataLoader(train_data, batch_size=int(config["batch_size"]), shuffle=True, pin_memory=True)

    if config["use_meta_semi"] == "True" or config["use_semi"] == "True" or config["use_col_enhance"] == "True":
        semi_data = SQLDataset(config["semi_data_path"], config, featurizer, False)

    num_samples = len(train_data)
    config["num_train_steps"] = int(num_samples * int(config["epochs"]) / int(config["batch_size"]))
    step_per_epoch = num_samples / int(config["batch_size"])
    print("total_steps: {0}, warm_up_steps: {1}".format(config["num_train_steps"], config["num_warmup_steps"]))

    model = create_model(config, is_train=True)

    evaluator = HydraEvaluator(model_path, config, featurizer, model, note)
    print("start training")
    epoch = 0
    cur_acc = 0.0
    best_acc = 0.0
    best_epoch = -1

    while True:
        config["state"] = "train"
        if (config["use_meta_semi"] == "True" or config["use_semi"] == "True" or config["use_col_enhance"] == "True") and cur_acc >= float(config["semi_theshold"]):
            model.predict_for_unlabeled_data(semi_data)

        #meta-semi-----------------------------------------------------------------------------------
        if config["use_meta_semi"] == "True" and cur_acc >= float(config["semi_theshold"]):
            print("meta-semi")

            task_data = train_data.get_meta_semi_task(semi_data)
            meta_semi_step = 0
            for tasks in task_data:
                for task in tasks:
                    cur_loss, flag_list = model.train_on_meta_semi_task(task)
                    currentDT = datetime.datetime.now()
                    meta_semi_step += 1
                    print("[{3}] epoch {0}, task_step {1}, task_loss={2:.4f}, task_flag={4}".format(epoch, meta_semi_step, cur_loss, currentDT.strftime("%m-%d %H:%M:%S"), str(flag_list)))

        # meta-----------------------------------------------------------------------------------
        if config["use_col_enhance"] == "True" and cur_acc >= float(config["semi_theshold"]):
            print("col_enhance")
            config["state"] = "train_col"
            task_data = train_data.get_col_enhance_task(semi_data)
            meta_step = 0
            for i, task in enumerate(task_data):
                cur_loss = model.train_on_column_force(task)
                currentDT = datetime.datetime.now()
                meta_step += 1
                if i % 10 == 0 and i > 0:
                    print("[{3}] epoch {0}, meta_step {1}, batch_loss={2:.4f}".format(epoch, meta_step, cur_loss,
                                                                              currentDT.strftime("%m-%d %H:%M:%S")))
            config["state"] = "train"

        if config["use_semi"] == "True" and cur_acc >= float(config["semi_theshold"]):
            train_model_inputs = train_data.get_semi_data(semi_data)
            train_data_loader = torch_data.DataLoader(train_model_inputs, batch_size=int(config["batch_size"]), shuffle=True, pin_memory=True)
            print(len(train_model_inputs))
        else:
            train_data_loader = torch_data.DataLoader(train_data, batch_size=int(config["batch_size"]), shuffle=True, pin_memory=True)
            print(len(train_data))

        loss_avg, step = 0.0, 0
        for batch_id, batch in enumerate(train_data_loader):
            # print(batch_id)
            cur_loss = model.train_on_batch(batch)
            loss_avg = (loss_avg * step + cur_loss) / (step + 1)
            step += 1
            if batch_id % 100 == 0:
                currentDT = datetime.datetime.now()
                print("[{3}] epoch {0}, batch {1}, batch_loss={2:.4f}".format(epoch, batch_id, cur_loss, currentDT.strftime("%m-%d %H:%M:%S")))


        if args.note:
            print(args.note)
        # model.save(model_path, epoch)
        cur_acc = evaluator.eval(epoch)
        if cur_acc >= best_acc:
            # if best_epoch != -1:
            #     os.remove(os.path.join(model_path, "model_{}.pt".format(best_epoch)))
            best_acc = cur_acc
            best_epoch = epoch
        model.save(model_path, epoch)
        epoch += 1
        if epoch >= int(config["epochs"]):
            break

else:
    raise Exception("Job type {0} is not supported for now".format(args.job))