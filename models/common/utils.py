import sys
import torch
import random
import os
import numpy as np
import json
import pickle
import random
import hashlib
import logging
from datetime import datetime


def dump_final_results(params, eval_results, model):
    result_str = "\t".join(["{}-{:.4f}".format(k, v) for k, v in eval_results.items()])

    key_info = [
        "dataset",
        "train_anomaly_ratio",
        "feature_type",
        "label_type",
        "use_attention",
    ]

    args = sys.argv
    model_name = args[0].replace("_demo.py", "")
    args = args[1:]
    input_params = [
        "{}:{}".format(args[idx * 2].strip("--"), args[idx * 2 + 1])
        for idx in range(len(args) // 2)
    ]
    recorded_params = ["{}:{}".format(k, v) for k, v in params.items() if k in key_info]

    params_str = "\t".join(input_params + recorded_params)

    with open(os.path.join(f"{params['dataset']}.txt"), "a+") as fw:
        info = "{} {} {} {} {} train: {:.3f} test: {:.3f}\n".format(
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            params["hash_id"],
            model_name,
            params_str,
            result_str,
            model.time_tracker["train"],
            model.time_tracker["test"],
        )
        fw.write(info)


def dump_params(params):
    hash_id = hashlib.md5(
        str(sorted([(k, v) for k, v in params.items()])).encode("utf-8")
    ).hexdigest()[0:8]
    params["hash_id"] = hash_id
    save_dir = os.path.join("./experiment_records", hash_id)
    os.makedirs(save_dir, exist_ok=True)

    json_pretty_dump(params, os.path.join(save_dir, "params.json"))

    log_file = os.path.join(save_dir, hash_id + ".log")
    # logs will not show in the file without the two lines.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        # handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        handlers=[logging.FileHandler(log_file)]
    )

    logging.info(json.dumps(params, indent=4))
    return save_dir


def decision(probability):
    return random.random() < probability


def json_pretty_dump(obj, filename):
    with open(filename, "w") as fw:
        json.dump(
            obj,
            fw,
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
            ensure_ascii=False,
        )


def tensor2flatten_arr(tensor):
    return tensor.data.cpu().numpy().reshape(-1)


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    # 设置CPU生成随机数的种子，方便下次复现实验结果。
    torch.manual_seed(seed)


def set_device(gpu=-1):
    # device_count = torch.cuda.device_count()
    # print(device_count)
    if gpu != -1 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu))
    else:
        device = torch.device("cpu")
    return device


def dump_pickle(obj, file_path):
    logging.info("Dumping to {}".format(file_path))
    with open(file_path, "wb") as fw:
        pickle.dump(obj, fw)


def load_pickle(file_path):
    logging.info("Loading from {}".format(file_path))
    with open(file_path, "rb") as fr:
        return pickle.load(fr)

