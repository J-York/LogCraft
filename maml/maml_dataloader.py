"""
The interface to load log datasets. The datasets currently supported include
HDFS and BGL.

Authors:
    LogPAI Team

"""

import logging
import pandas as pd
import os
import numpy as np
import re
import pickle
import json
from collections import OrderedDict, defaultdict
from torch.utils.data import Dataset

import collections


def load_sessions(data_dir):
    # with open(os.path.join(data_dir, "data_desc.json"), "r") as fr:
    #     data_desc = json.load(fr)
    with open(os.path.join(data_dir, "session_train.pkl"), "rb") as fr:
        session_train = pickle.load(fr)
    with open(os.path.join(data_dir, "session_test.pkl"), "rb") as fr:
        session_test = pickle.load(fr)

    train_labels = [
        v["label"] if not isinstance(v["label"], list) else int(sum(v["label"]) > 0)
        for _, v in session_train.items()
    ]
    test_labels = [
        v["label"] if not isinstance(v["label"], list) else int(sum(v["label"]) > 0)
        for _, v in session_test.items()
    ]

    num_train = len(session_train)
    ratio_train = sum(train_labels) / num_train
    num_test = len(session_test)
    ratio_test = sum(test_labels) / num_test
    logging.info("Load from {}".format(data_dir))
    # logging.info(json.dumps(data_desc, indent=4))
    logging.info(
        "# train sessions {} ({:.2f} anomalies)".format(num_train, ratio_train)
    )
    logging.info("# test sessions {} ({:.2f} anomalies)".format(num_test, ratio_test))
    return session_train, session_test

def load_windows(data_dir,save_name,templates_path,train_lines=1000,load_part=False):
    id_path=f"{data_dir}/{save_name}.npy"
    label_path=f"{data_dir}/{save_name}_label.npy"
    next_path=f"{data_dir}/{save_name}_next.npy"
    id_array=np.load(id_path)
    label_array=np.load(label_path)
    next_array=np.load(next_path)
    total_windows = len(id_array)  # 假设 id_array 是窗口的标识数组
    train_lines = int(0.8 * total_windows)
    train_num=0
    window_idx=list(range(len(id_array)))
    np.random.shuffle(window_idx)
    window_idx_train=[]
    window_idx_test=[]
    for idx in window_idx:
        if label_array[idx]>0:
            window_idx_test.append(idx)
        elif train_num<train_lines:
            train_num+=1
            window_idx_train.append(idx)
        else:
            window_idx_test.append(idx)
    id_train=id_array[window_idx_train]
    label_train=label_array[window_idx_train]
    next_train=next_array[window_idx_train]
    id_test=id_array[window_idx_test]
    label_test=label_array[window_idx_test]
    next_test=next_array[window_idx_test]
    templates = list(pd.read_csv(templates_path, engine="c", na_filter=False, memory_map=True)["EventTemplate"])
    train={"features":id_train,"window_labels":next_train,"window_anomalies":label_train}
    test={"features":id_test,"window_labels":next_test,"window_anomalies":label_test}
    return train,test,templates

# def load_windows(data_dir, save_name, templates_path, train_ratio=0.8, load_part=False):
#     id_path = f"{data_dir}/{save_name}.npy"
#     label_path = f"{data_dir}/{save_name}_label.npy"
#     next_path = f"{data_dir}/{save_name}_next.npy"

#     id_array = np.load(id_path)
#     label_array = np.load(label_path)
#     next_array = np.load(next_path)

#     window_idx = list(range(len(id_array)))
#     normal_idx = [idx for idx in window_idx if label_array[idx] == 0]
#     anomaly_idx = [idx for idx in window_idx if label_array[idx] > 0]

#     np.random.shuffle(normal_idx)

#     # 计算训练集大小
#     train_size = int(len(normal_idx) * train_ratio)
#     window_idx_train = normal_idx[:train_size]
#     window_idx_test = normal_idx[train_size:] + anomaly_idx

#     id_train = id_array[window_idx_train]
#     label_train = label_array[window_idx_train]
#     next_train = next_array[window_idx_train]

#     id_test = id_array[window_idx_test]
#     label_test = label_array[window_idx_test]
#     next_test = next_array[window_idx_test]

#     templates = list(pd.read_csv(templates_path, engine="c", na_filter=False, memory_map=True)["EventTemplate"])

#     train = {"features": id_train, "window_labels": next_train, "window_anomalies": label_train}
#     test = {"features": id_test, "window_labels": next_test, "window_anomalies": label_test}

#     return train, test, templates

class log_zte_dataset(Dataset):
    # def __init__(self, session_dict,k_shot, k_query,feature_type="semantics"):
    def __init__(self, window_dict,templates,feature_type="semantics"):

        # self.k_shot = k_shot  # k-shot
        # self.k_query = k_query  # for evaluation
        # self.setsz = self.n_way * self.k_shot  # num of samples per set
        # self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.templates=templates
        flatten_data_list = []
        features = window_dict["features"]
        window_labels = window_dict["window_labels"]
        window_anomalies = window_dict["window_anomalies"]
        for window_idx in range(len(window_labels)):
            sample = {
                "session_idx": window_idx,  # not session id
                "features": features[window_idx],
                "window_labels": window_labels[window_idx],
                "window_anomalies": window_anomalies[window_idx],
                # "templates":templates,
            }
            flatten_data_list.append(sample)
        # print(flatten_data_list[0]["features"])
        self.flatten_data_list = flatten_data_list

    def __len__(self):
        return len(self.flatten_data_list)
    
    def __getitem__(self, idx):
        return self.flatten_data_list[idx]


class log_maml_dataset(Dataset):
    # def __init__(self, session_dict,k_shot, k_query,feature_type="semantics"):
    def __init__(self, session_dict,templates,feature_type="semantics"):

        # self.k_shot = k_shot  # k-shot
        # self.k_query = k_query  # for evaluation
        # self.setsz = self.n_way * self.k_shot  # num of samples per set
        # self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.templates=templates
        flatten_data_list = []
        # print(list(session_dict.keys())[0])
        # session_dict=collections.OrderedDict(session_dict.items())
        # print(type(session_dict))
        # print(list(session_dict.keys())[0])
        # key=list(session_dict.keys())[0]
        # print(session_dict[key]["features"])
        # flatten all sessions
        for session_idx, data_dict in enumerate(session_dict.values()):
            features = data_dict["features"][feature_type]
            window_labels = data_dict["window_labels"]
            window_anomalies = data_dict["window_anomalies"]
            # <class 'numpy.ndarray'>
            # <class 'list'>
            # <class 'list'>
            # print(type(features))
            # print(type(window_labels))
            # print(type(window_anomalies))
            # print(11111111)
            for window_idx in range(len(window_labels)):
                sample = {
                    "session_idx": session_idx,  # not session id
                    "features": features[window_idx],
                    "window_labels": window_labels[window_idx],
                    "window_anomalies": window_anomalies[window_idx],
                    # "templates":templates,
                }
                flatten_data_list.append(sample)
        # print(flatten_data_list[0]["features"])
        self.flatten_data_list = flatten_data_list

    def __len__(self):
        return len(self.flatten_data_list)
    
    def __getitem__(self, idx):
        return self.flatten_data_list[idx]

    # def __getitem__(self, idx):
    #     support=dict()
    #     query=dict()
    #     selected_idx = np.random.choice(len(self.flatten_data_list), self.k_shot + self.k_query, False)
    #     np.random.shuffle(selected_idx)
    #     indexDtrain = np.array(selected_idx[:self.k_shot])  # idx for Dtrain
    #     indexDtest = np.array(selected_idx[self.k_shot:])  # idx for Dtest
    #     support_session_idx=[self.flatten_data_list[index]["session_idx"] for index in indexDtrain]
    #     support_features=[list(self.flatten_data_list[index]["features"]) for index in indexDtrain]
    #     support_window_labels=[self.flatten_data_list[index]["window_labels"] for index in indexDtrain]
    #     support_window_anomalies=[self.flatten_data_list[index]["window_anomalies"] for index in indexDtrain]
    #     support["session_idx"]=np.array(support_session_idx)
    #     support["features"]=np.array(support_features)
    #     support["window_labels"]=np.array(support_window_labels)
    #     support["window_anomalies"]=np.array(support_window_anomalies)

    #     query_session_idx = [self.flatten_data_list[index]["session_idx"] for index in indexDtest]
    #     query_features = [list(self.flatten_data_list[index]["features"]) for index in indexDtest]
    #     query_window_labels = [self.flatten_data_list[index]["window_labels"] for index in indexDtest]
    #     query_window_anomalies = [self.flatten_data_list[index]["window_anomalies"] for index in indexDtest]
    #     query["session_idx"] = np.array(query_session_idx)
    #     query["features"] = np.array(query_features)
    #     query["window_labels"] = np.array(query_window_labels)
    #     query["window_anomalies"] = np.array(query_window_anomalies)
    #     # support = np.array(self.flatten_data_list)[indexDtrain].tolist()  # get all images filename for current Dtrain
    #     # query=np.array(self.flatten_data_list)[indexDtest].tolist()
    #     return support,query
