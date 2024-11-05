import json
import pickle
import pandas as pd
import logging
import setproctitle
import torch
import sys
import os
import requests
from time import perf_counter
sys.path.append("../")

import argparse

from models import DeepLog, LogAnomaly
from models import Transformer
from models.common.preprocess import FeatureExtractor
# from models.DeepLog.deeploglizer.common.dataloader import load_sessions, log_dataset, load_windows
from models.common.utils import seed_everything
from maml.maml_dataloader import load_sessions,log_maml_dataset

from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )

##### Model params
parser.add_argument("--model_name", default="LSTM", type=str)
parser.add_argument("--use_attention", action="store_true")
parser.add_argument("--hidden_size", default=128, type=int)
parser.add_argument("--num_layers", default=2, type=int)
parser.add_argument("--num_directions", default=2, type=int)
parser.add_argument("--embedding_dim", default=32, type=int)

##### Dataset params
parser.add_argument("--dataset", default="BGL", type=str)
parser.add_argument(
    "--data_dir", default="../data/processed/Spirit/spirit_0.30_60_random_tar", type=str
)
parser.add_argument("--window_size", default=20, type=int)
# stride步长
parser.add_argument("--stride", default=1, type=int)

##### Input params
parser.add_argument("--feature_type", default="sequentials", type=str, choices=["sequentials", "semantics"])
parser.add_argument("--label_type", default="next_log", type=str)
parser.add_argument("--eval_type",default="window",type=str)
parser.add_argument("--use_tfidf", action="store_true")
parser.add_argument("--max_token_len", default=50, type=int)
parser.add_argument("--min_token_count", default=1, type=int)
# Uncomment the following to use pretrained word embeddings. The "embedding_dim" should be set as 300
# parser.add_argument(
#     "--pretrain_path", default="../data/pretrain/wiki-news-300d-1M.vec", type=str
# )

##### Training params
parser.add_argument("--epoches", default=50, type=int)
parser.add_argument("--batch_size", default=512, type=int)
parser.add_argument("--learning_rate", default=0.00001, type=float)
parser.add_argument("--topk", default=5, type=int)
parser.add_argument("--patience", default=5, type=int)

##### Others
parser.add_argument("--random_seed", default=42, type=int)
parser.add_argument("--gpu", default=0, type=int)

params = vars(parser.parse_args())

def initial_model(config_dict:dict):
    model_name=config_dict["model"]
    del config_dict["model"]
    del config_dict["epoches"]
    configs=config_dict
    if model_name=="DeepLog":
        for key,value in configs.items():
            params[key]=value
        model=DeepLog(dict({"num_labels":config_dict["vocab_size"],"vocab_size":config_dict["vocab_size"]}),**params)
        # for param in model.embedder.embedding_layer.parameters():
        #     param.requires_grad = False
    elif model_name=="Transformer":
        for key,value in configs.items():
            params[key]=value
        # model = TransformerClassifier(num_classes=2, vocab_size=500, max_len=configs["window_size"], d_model=384, nhead = 8, num_layers=1, dropout=0.2, device=config_dict["device"])
        model = Transformer(dict({"num_labels":config_dict["vocab_size"],"vocab_size":config_dict["vocab_size"]}),**params)
    elif model_name=="LogAnomaly":
        for key,value in configs.items():
            params[key]=value
        model=LogAnomaly(dict({"num_labels":config_dict["vocab_size"],"vocab_size":config_dict["vocab_size"]}),**params)
    return model


if __name__ == '__main__':
    params["gpu"]=1
    params["learning_rate"]=0.001
    params["topk"]=10

    # change data name here
    data_name = params["dataset"]
    params["data_dir"]=os.path.join('../data_preprocessed/', data_name)
    vocab_path = os.path.join(params["data_dir"], 'vocab.json')
    print_result_path=data_name+'_run_result.txt'
    params["window_size"] = 20
    params["stride"] = 20
    params["batch_size"] = 64
    seed_everything(params["random_seed"])
    # params['eval_type'] = "window"
    config_dict={"model":"Transformer",
        "configs":{"window_size":20,
        "num_layers":2,
        "stride":20,
        "embedding_dim":300,
        "hidden_size":256,
        "pretrain":True
        },
        "device":params["gpu"]}
    
    # model.load_state_dict(torch.load('../middle/pretrained_models/transformer_tb.pth'))

    session_train, session_test = load_sessions(data_dir=params["data_dir"])
    ext = FeatureExtractor(vocab_path=vocab_path, **params)

    session_train = ext.fit_transform(session_train)
    session_test = ext.transform(session_test, datatype="test")


    dataset_train = log_maml_dataset(session_train, ext.id2log_train, feature_type=params["feature_type"])
    dataset_test = log_maml_dataset(session_test, ext.id2log_train, feature_type=params["feature_type"])

    dataloader_train = DataLoader(
        dataset_train, batch_size=params["batch_size"], shuffle=True, pin_memory=True
    )
    dataloader_test = DataLoader(
        dataset_test, batch_size=512, shuffle=False, pin_memory=True
    )

    templates_list = list(ext.id2log_train.values())

    config_dict["vocab_size"] = max(ext.log2id_train.values()) + 1
    print("vocab size: ", config_dict["vocab_size"])

    stime = perf_counter()

    model=initial_model(config_dict)

    eval_results = model.fit(
        dataloader_train,
        test_loader=dataloader_test,
        epoches=params["epoches"],
        learning_rate=params["learning_rate"],
        templates=ext.id2log_train,
        print_result_path=print_result_path
    )

    etime = perf_counter()
    print(f'elapsed_time: {etime - stime} s')

    # df_before = pd.DataFrame.from_dict(before_eval_results)
    # before_save_dir = f'./test_results/before_results_dt'
    # os.makedirs(before_save_dir, exist_ok=True)
    # df_before.to_csv(os.path.join(before_save_dir, f'before.csv'),index=0)
    print(eval_results)

def detect(data_name,config_dict):
    params["data_dir"]=os.path.join('../data_preprocessed/', data_name)
    vocab_path = os.path.join(params["data_dir"], 'vocab.json')
    # params["pretrain"]=False
    print_result_path=data_name+'_run_result.txt'
    seed_everything(params["random_seed"])
    session_train, session_test = load_sessions(data_dir=params["data_dir"])
    ext = FeatureExtractor(vocab_path=vocab_path, **params)

    session_train = ext.fit_transform(session_train)
    session_test = ext.transform(session_test, datatype="test")


    dataset_train = log_maml_dataset(session_train, ext.id2log_train, feature_type=params["feature_type"])
    dataset_test = log_maml_dataset(session_test, ext.id2log_train, feature_type=params["feature_type"])

    dataloader_train = DataLoader(
        dataset_train, batch_size=params["batch_size"], shuffle=True, pin_memory=True
    )
    dataloader_test = DataLoader(
        dataset_test, batch_size=512, shuffle=False, pin_memory=True
    )

    templates_list = list(ext.id2log_train.values())

    config_dict["vocab_size"] = max(ext.log2id_train.values()) + 1
    print("vocab size: ", config_dict["vocab_size"])

    stime = perf_counter()

    model=initial_model(config_dict)

    eval_results = model.fit(
        dataloader_train,
        test_loader=dataloader_test,
        epoches=params["epoches"],
        learning_rate=params["learning_rate"],
        templates=ext.id2log_train,
        print_result_path=print_result_path
    )

    etime = perf_counter()
    print(f'elapsed_time: {etime - stime} s')

    # df_before = pd.DataFrame.from_dict(before_eval_results)
    # before_save_dir = f'./test_results/before_results_dt'
    # os.makedirs(before_save_dir, exist_ok=True)
    # df_before.to_csv(os.path.join(before_save_dir, f'before.csv'),index=0)
    print(eval_results)
    
