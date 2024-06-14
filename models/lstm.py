import json
import math
import torch
from torch import nn
import logging
import time
from tqdm import tqdm
import sys

from models.base_model import ForcastBasedModel,ForcastBasedModel_wos
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from models.common.utils import tensor2flatten_arr
import pandas as pd
import numpy as np
from sentence_transformers import util

class Attention(nn.Module):
    def __init__(self, input_size, max_seq_len):
        super(Attention, self).__init__()
        self.atten_w = nn.Parameter(torch.randn(max_seq_len, input_size, 1))
        self.atten_bias = nn.Parameter(torch.randn(max_seq_len, 1, 1))
        self.glorot(self.atten_w)
        self.zeros(self.atten_bias)

    def forward(self, lstm_input):
        input_tensor = lstm_input.transpose(1, 0)  # f x b x d

        # bmm：矩阵乘法
        input_tensor = (
            torch.bmm(input_tensor, self.atten_w) + self.atten_bias
        )  # f x b x out
        input_tensor = input_tensor.transpose(1, 0)
        atten_weight = input_tensor.tanh()

        weighted_sum = torch.bmm(atten_weight.transpose(1, 2), lstm_input).squeeze()

        return weighted_sum

    def glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

    def zeros(self, tensor):
        if tensor is not None:
            tensor.data.fill_(0)


class LSTM(ForcastBasedModel):
    def __init__(
        self,
        meta_data,
        hidden_size=100,
        num_directions=2,
        num_layers=1,
        window_size=None,
        use_attention=False,
        embedding_dim=16,
        model_save_path="./lstm_models",
        feature_type="sequentials",
        label_type="next_log",
        eval_type="session",
        topk=5,
        use_tfidf=False,
        freeze=False,
        gpu=-1,
        pretrain=False,
        **kwargs
    ):
        super().__init__(
            meta_data=meta_data,
            model_save_path=model_save_path,
            feature_type=feature_type,
            label_type=label_type,
            eval_type=eval_type,
            topk=topk,
            use_tfidf=use_tfidf,
            embedding_dim=embedding_dim,
            freeze=freeze,
            gpu=gpu,
        )
        num_labels = meta_data["num_labels"]
        self.feature_type = feature_type
        self.label_type = label_type
        self.hidden_size = hidden_size
        self.num_directions = num_directions
        self.window_size = window_size
        self.use_attention = use_attention
        self.use_tfidf = use_tfidf
        self.embedding_dim = embedding_dim
        self.pretrain = False
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=(self.num_directions == 2),
        )
        if self.use_attention:
            assert window_size is not None, "window size must be set if use attention"
            self.attn = Attention(hidden_size * num_directions, window_size)
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion=nn.MSELoss(reduction='sum')
        self.prediction_layer = nn.Linear(
            self.hidden_size * self.num_directions, embedding_dim
        )
        

    def maml_train(self,input_dict,templates=None):
        return self.forward(input_dict,templates)

    def forward(self, input_dict,templates=None):
        if self.label_type == "anomaly":
            y = input_dict["window_anomalies"].long().view(-1)
        elif self.label_type == "next_log":
            y = input_dict["window_labels"].long().view(-1)
        self.batch_size = y.size()[0]
        x = input_dict["features"]
        x = self.embedder(x,templates).to(self.device)
        if self.feature_type == "semantics":
            if not self.use_tfidf:
                x = x.sum(dim=-2)  # add tf-idf

        outputs, _ = self.rnn(x.float())

        if self.use_attention:
            representation = self.attn(outputs)
        else:
            # representation = outputs.mean(dim=1)
            representation = outputs[:, -1, :]

        logits = self.prediction_layer(representation)
        # y_pred = logits.softmax(dim=-1)
        y=self.embedder(y,templates).to(self.device)
        loss = self.criterion(logits, y)
        return_dict = {"loss": loss, "y_pred": logits}
        return return_dict
    
    def fit(self, train_loader, test_loader=None, epoches=10, learning_rate=1.0e-3,templates=None):
        self.to(self.device)
        logging.info(
            "Start training on {} batches with {}.".format(
                len(train_loader), self.device
            )
        )
        best_f1 = -float("inf")
        best_results = None
        worse_count = 0
        results=list()
        self.time_tracker["train"]=0.0
        for epoch in range(1, epoches + 1):
            epoch_time_start = time.time()
            model = self.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            batch_cnt = 0
            epoch_loss = 0
            tbar = tqdm(train_loader, desc="\r")
            num_batch = len(train_loader)
            for i, batch_input in enumerate(tbar):
                loss = self.maml_train(self.__input2device(batch_input),templates)["loss"]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
                batch_cnt += 1
                # if not self.pretrain and test_loader is not None and (epoch % 1 == 0):
                # # best_results=self.evaluate(test_loader)
                #     eval_results = self.maml_evaluate(test_loader,templates)
                #     # if eval_results["f1"] > best_f1:
                #     # best_f1 = eval_results["f1"]
                #     best_results = eval_results
                #     best_results["epoches"] = int(epoch)
                #     best_results["train_loss"] = loss
                #     print(best_results)
                #     sys.stdout.flush()
                # #     results.append(best_results)
                tbar.set_description("Train loss: %.5f" % (epoch_loss / (i + 1)))
            epoch_loss = epoch_loss / batch_cnt
            epoch_time_elapsed = time.time() - epoch_time_start
            logging.info(
                "Epoch {}/{}, training loss: {:.5f} [{:.2f}s]".format(epoch, epoches, epoch_loss, epoch_time_elapsed)
            )
            self.time_tracker["train"] += epoch_time_elapsed

            if test_loader is not None and (epoch % 1 == 0):
                # best_results=self.evaluate(test_loader)
                eval_results = self.maml_evaluate(test_loader,templates)
                # if eval_results["f1"] > best_f1:
                # best_f1 = eval_results["f1"]
                best_results = eval_results
                best_results["epoches"] = int(epoch)
                # print(best_results)
                results.append(best_results)
                # print(results)
                    # self.save_model()
                    # worse_count = 0
                # else:
                #     worse_count += 1
                #     if worse_count >= self.patience:
                #         logging.info("Early stop at epoch: {}".format(epoch))
                #         break
        # best_results["train_time"]=self.time_tracker['train']
        # self.load_model(self.model_save_file)
        # return best_results
        return results


    def maml_evaluate(self,test_loader,templates=None):
        loss=0
        with torch.no_grad():
            y_pred = []
            store_dict = defaultdict(list)
            templates_encoding=self.embedder.embedding_layer.encode(templates,show_progress_bar=False,convert_to_tensor=True,device=self.device)
            for batch_input in test_loader:
                # if templates is not None:
                #     batch_input["templates"]=torch.from_numpy(np.array(templates))
                # print(batch_input)
                return_dict = self.maml_train(self.__input2device(batch_input),templates)
                y_pred = return_dict["y_pred"]
                loss+=return_dict["loss"].item()
                y_pred_scores=[]
                for pred in y_pred:
                    cosine_scores = util.cos_sim(pred, templates_encoding)
                    y_pred_scores.append(cosine_scores)
                y_pred_scores=torch.cat(y_pred_scores,dim=0)

                # torch.topk:返回的第一个为对应y_pred的具体值,第二个为在y_pred中的索引
                y_prob_topk, y_pred_topk = torch.topk(y_pred_scores, self.topk)  # b x topk

                store_dict["session_idx"].extend(
                    tensor2flatten_arr(batch_input["session_idx"])
                )
                store_dict["window_anomalies"].extend(
                    tensor2flatten_arr(batch_input["window_anomalies"])
                )
                store_dict["window_labels"].extend(
                    tensor2flatten_arr(batch_input["window_labels"])
                )
                store_dict["x"].extend(batch_input["features"].data.cpu().numpy())
                store_dict["y_pred_topk"].extend(y_pred_topk.data.cpu().numpy())
                store_dict["y_prob_topk"].extend(y_prob_topk.data.cpu().numpy())

            store_df = pd.DataFrame(store_dict)
            best_result = None
            # float("inf"):正无穷
            best_f1 = -float("inf")

            # print(store_dict["y_prob_topk"])
            # # 添加topp
            # for i in range(len(store_dict["y_prob_topk"])):
            #     for k in range(self.topk):
            #         if store_dict["y_prob_topk"][i][k]<min_p:
            #             store_dict["y_pred_topk"][i][k]=-1
            topkdf = pd.DataFrame(store_df["y_pred_topk"].tolist())
            hit_df = pd.DataFrame()
            for col in sorted(topkdf.columns):
                topk = col + 1
                hit = (topkdf[col] == store_df["window_labels"]).astype(int)
                hit_df[topk] = hit
                if col == 0:
                    acc_sum = 2 ** topk * hit
                else:
                    acc_sum += 2 ** topk * hit  # acc_num是一串数据series，长度为窗口总数（测试样本数），每个位数的数等于 2 ** k，k为该样本的y在pred中的索引
            # 将0处的（即pred的topk均没有y的测试样本）取很大的值，使得（转2）
            acc_sum[acc_sum == 0] = 2 ** (1 + len(topkdf.columns))
            hit_df["acc_num"] = acc_sum

            for col in sorted(topkdf.columns):
                topk = col + 1
                check_num = 2 ** topk
                # 2：使得 hit_df["acc_num"] > check_num 时，不是y在pred的索引在当前k（col）之后（前k个没有预测到y），就是整个topk都没有y，均为当前的异常情况
                # ~：0变1,1变0,~x == -x-1
                store_df["window_pred_anomaly_{}".format(topk)] = (
                    ~(hit_df["acc_num"] <= check_num)
                ).astype(int) # 若在k处命中y，k之前均为1（异常），k之后均为0（正常）
            # store_df.to_csv("store_{}_2.csv".format(dtype), index=False)


            if self.eval_type == "session":
                use_cols = ["session_idx", "window_anomalies"] + [
                    f"window_pred_anomaly_{topk}" for topk in range(1, self.topk + 1)
                ]
                # 分组后，sum将每一个session_idx的各列相加
                session_df = (
                    store_df[use_cols].groupby("session_idx", as_index=False).sum()
                )
            else:
                session_df = store_df
            # session_df.to_csv("session_{}_2.csv".format(dtype), index=False)

            for topk in range(1, self.topk + 1):
                pred = (session_df[f"window_pred_anomaly_{topk}"] > 0).astype(int)
                y = (session_df["window_anomalies"] > 0).astype(int)
                # pred_0=(session_df[f"window_pred_anomaly_{topk}"] == 0).astype(int)
                # y_0=(session_df["window_anomalies"] == 0).astype(int).sum()
                # correct_0=(session_df[f"window_pred_anomaly_{topk}"] == 0 and session_df["window_anomalies"] == 0).astype(int).sum()
                # y_0 = len(y)-y.sum()
                # correct_0 = (y.to_frame() == 0 and pred.to_frame() == 0).astype(int).sum()
                # y_0 = 0
                # correct_0 = 0
                # for i in y.keys():
                #     if y[i] == 0:
                #         y_0 += 1
                #         if pred[i] == 0:
                #             correct_0 += 1
                # window_topk_acc = 1 - store_df["window_anomalies"].sum() / len(store_df)
                tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
                eval_results = {
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "TN": tn,
                    "recall": recall_score(y, pred),
                    "precision": precision_score(y, pred),
                    "f1s": f1_score(y, pred),
                    "acc":accuracy_score(y, pred),
                }
                if eval_results["f1s"] >= best_f1:
                    best_result = eval_results
                    best_f1 = eval_results["f1s"]
            best_result["loss"]=loss
            print(f'TP:{best_result["TP"]}, FP:{best_result["FP"]}, TN:{best_result["TN"]}, FN:{best_result["FN"]}')
            print(f'precision:{best_result["precision"]}, recall:{best_result["recall"]}, f1s:{best_result["f1s"]}')
            return best_result
        
    def __input2device(self, batch_input):
        return {k: v.to(self.device) for k, v in batch_input.items()}
    
class LSTM_tokenpredict(ForcastBasedModel):
    def __init__(
        self,
        meta_data,
        hidden_size=100,
        num_directions=2,
        num_layers=4,
        window_size=None,
        use_attention=False,
        embedding_dim=32,
        model_save_path="./lstm_models",
        feature_type="sequentials",
        label_type="next_log",
        eval_type="window",
        topk=5,
        use_tfidf=False,
        freeze=False,
        gpu=-1,
        pretrain=False,
        **kwargs
    ):
        super().__init__(
            meta_data=meta_data,
            model_save_path=model_save_path,
            feature_type=feature_type,
            label_type=label_type,
            eval_type=eval_type,
            topk=topk,
            use_tfidf=use_tfidf,
            embedding_dim=embedding_dim,
            freeze=freeze,
            gpu=gpu,
        )
        num_labels = meta_data["num_labels"]
        self.feature_type = feature_type
        self.label_type = label_type
        self.hidden_size = hidden_size
        self.num_directions = num_directions
        self.window_size = window_size
        self.use_attention = use_attention
        self.use_tfidf = use_tfidf
        self.sbert_dim = 384
        self.embedding_dim = embedding_dim
        self.pretrain = pretrain
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=(self.num_directions == 2),
        )
        if self.use_attention:
            assert window_size is not None, "window size must be set if use attention"
            self.attn = Attention(hidden_size * num_directions, window_size)
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion=nn.MSELoss(reduction='sum')
        self.prediction_layer = nn.Linear(
            self.hidden_size * self.num_directions, meta_data["vocab_size"]
        )

    def maml_train(self,input_dict,templates=None):
        return self.forward(input_dict,templates)

    def forward(self, input_dict,templates=None):
        if self.label_type == "anomaly":
            y = input_dict["window_anomalies"].long().view(-1)
        elif self.label_type == "next_log":
            y = input_dict["window_labels"].long().view(-1)
        self.batch_size = y.size()[0]
        x = input_dict["features"]
        x = self.embedder(x,templates).to(self.device)
        # if self.feature_type == "semantics":
        #     if not self.use_tfidf:
        #         x = x.sum(dim=-2)  # add tf-idf
        outputs, _ = self.rnn(x.float())

        if self.use_attention:
            representation = self.attn(outputs)
        else:
            # representation = outputs.mean(dim=1)
            representation = outputs[:, -1, :]

        logits = self.prediction_layer(representation)
        y_pred = logits.softmax(dim=-1)
        # y=self.embedder(y,templates).to(self.device)
        y = input_dict["window_labels"].long().view(-1)
        loss = self.criterion(logits, y)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict
    
    def fit(self, train_loader, test_loader=None, epoches=10, learning_rate=1.0e-3,templates=None):
        self.to(self.device)
        logging.info(
            "Start training on {} batches with {}.".format(
                len(train_loader), self.device
            )
        )
        best_f1 = -float("inf")
        best_results = None
        worse_count = 0
        results=list()
        self.time_tracker["train"]=0.0
        for epoch in range(1, epoches + 1):
            epoch_time_start = time.time()
            model = self.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            batch_cnt = 0
            epoch_loss = 0
            tbar = tqdm(train_loader, desc="\r")
            num_batch = len(train_loader)
            for i, batch_input in enumerate(tbar):
                loss = self.maml_train(self.__input2device(batch_input),templates)["loss"]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
                batch_cnt += 1
                if not self.pretrain and test_loader is not None and (epoch % 1 == 0):
                # best_results=self.evaluate(test_loader)
                    eval_results = self.maml_evaluate(test_loader,templates)
                    # if eval_results["f1"] > best_f1:
                    # best_f1 = eval_results["f1"]
                    best_results = eval_results
                    best_results["epoches"] = int(epoch)
                    best_results["train_loss"] = loss
                    print(best_results)
                    sys.stdout.flush()
                #     results.append(best_results)
                tbar.set_description("Train loss: %.5f" % (epoch_loss / (i + 1)))
            epoch_loss = epoch_loss / batch_cnt
            epoch_time_elapsed = time.time() - epoch_time_start
            logging.info(
                "Epoch {}/{}, training loss: {:.5f} [{:.2f}s]".format(epoch, epoches, epoch_loss, epoch_time_elapsed)
            )
            self.time_tracker["train"] += epoch_time_elapsed

            # if not self.pretrain and test_loader is not None and (epoch % 1 == 0):
                # best_results=self.evaluate(test_loader)
            eval_results = self.maml_evaluate(test_loader,templates)
            # if eval_results["f1"] > best_f1:
            # best_f1 = eval_results["f1"]
            best_results = eval_results
            best_results["epoches"] = int(epoch)
            print(best_results)
            results.append(best_results)
        return results


    def maml_evaluate(self,test_loader,templates=None):
        loss=0
        with torch.no_grad():
            y_pred = []
            store_dict = defaultdict(list)
            # templates_encoding=self.embedder.embedding_layer.encode(templates,show_progress_bar=False,convert_to_tensor=True,device=self.device)
            for batch_input in test_loader:
                # if templates is not None:
                #     batch_input["templates"]=torch.from_numpy(np.array(templates))
                # print(batch_input)
                return_dict = self.maml_train(self.__input2device(batch_input),templates)
                y_pred = return_dict["y_pred"]
                loss+=return_dict["loss"].item()
                y_prob_topk, y_pred_topk = torch.topk(y_pred, self.topk)  # b x topk

                store_dict["session_idx"].extend(
                    tensor2flatten_arr(batch_input["session_idx"])
                )
                store_dict["window_anomalies"].extend(
                    tensor2flatten_arr(batch_input["window_anomalies"])
                )
                store_dict["window_labels"].extend(
                    tensor2flatten_arr(batch_input["window_labels"])
                )
                store_dict["x"].extend(batch_input["features"].data.cpu().numpy())
                store_dict["y_pred_topk"].extend(y_pred_topk.data.cpu().numpy())
                store_dict["y_prob_topk"].extend(y_prob_topk.data.cpu().numpy())

            store_df = pd.DataFrame(store_dict)
            best_result = None
            # float("inf"):正无穷
            best_f1 = -float("inf")

            topkdf = pd.DataFrame(store_df["y_pred_topk"].tolist())
            hit_df = pd.DataFrame()
            for col in sorted(topkdf.columns):
                topk = col + 1
                hit = (topkdf[col] == store_df["window_labels"]).astype(int)
                hit_df[topk] = hit
                if col == 0:
                    acc_sum = 2 ** topk * hit
                else:
                    acc_sum += 2 ** topk * hit  # acc_num是一串数据series，长度为窗口总数（测试样本数），每个位数的数等于 2 ** k，k为该样本的y在pred中的索引
            # 将0处的（即pred的topk均没有y的测试样本）取很大的值，使得（转2）
            acc_sum[acc_sum == 0] = 2 ** (1 + len(topkdf.columns))
            hit_df["acc_num"] = acc_sum

            for col in sorted(topkdf.columns):
                topk = col + 1
                check_num = 2 ** topk
                # 2：使得 hit_df["acc_num"] > check_num 时，不是y在pred的索引在当前k（col）之后（前k个没有预测到y），就是整个topk都没有y，均为当前的异常情况
                # ~：0变1,1变0,~x == -x-1
                store_df["window_pred_anomaly_{}".format(topk)] = (
                    ~(hit_df["acc_num"] <= check_num)
                ).astype(int) # 若在k处命中y，k之前均为1（异常），k之后均为0（正常）
            # store_df.to_csv("store_{}_2.csv".format(dtype), index=False)


            if self.eval_type == "session":
                use_cols = ["session_idx", "window_anomalies"] + [
                    f"window_pred_anomaly_{topk}" for topk in range(1, self.topk + 1)
                ]
                # 分组后，sum将每一个session_idx的各列相加
                session_df = (
                    store_df[use_cols].groupby("session_idx", as_index=False).sum()
                )
            else:
                session_df = store_df
            # session_df.to_csv("session_{}_2.csv".format(dtype), index=False)

            for topk in range(1, self.topk + 1):
                pred = (session_df[f"window_pred_anomaly_{topk}"] > 0).astype(int)
                y = (session_df["window_anomalies"] > 0).astype(int)
                tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
                eval_results = {
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "TN": tn,
                    "recall": recall_score(y, pred),
                    "precision": precision_score(y, pred),
                    "f1s": f1_score(y, pred),
                    "acc":accuracy_score(y, pred),
                }
                if eval_results["f1s"] >= best_f1:
                    best_result = eval_results
                    best_f1 = eval_results["f1s"]
            best_result["loss"]=loss
            print(f'TP:{best_result["TP"]}, FP:{best_result["FP"]}, TN:{best_result["TN"]}, FN:{best_result["FN"]}')
            print(f'precision:{best_result["precision"]}, recall:{best_result["recall"]}, f1s:{best_result["f1s"]}')
            return best_result
        
    def __input2device(self, batch_input):
        return {k: v.to(self.device) for k, v in batch_input.items()}

class DeepLog(ForcastBasedModel_wos):
    def __init__(
        self,
        meta_data,
        hidden_size=100,
        num_directions=2,
        num_layers=4,
        window_size=None,
        use_attention=False,
        embedding_dim=32,
        model_save_path="./lstm_models",
        feature_type="sequentials",
        label_type="next_log",
        eval_type="session",
        topk=5,
        use_tfidf=False,
        freeze=False,
        gpu=-1,
        pretrain=True,
        **kwargs
    ):
        super().__init__(
            meta_data=meta_data,
            model_save_path=model_save_path,
            feature_type=feature_type,
            label_type=label_type,
            eval_type=eval_type,
            topk=topk,
            use_tfidf=use_tfidf,
            embedding_dim=embedding_dim,
            freeze=freeze,
            gpu=gpu,
        )
        num_labels = meta_data["num_labels"]
        self.feature_type = feature_type
        self.label_type = label_type
        self.hidden_size = hidden_size
        self.num_directions = num_directions
        self.window_size = window_size
        self.use_attention = use_attention
        self.use_tfidf = use_tfidf
        self.sbert_dim = 384
        self.embedding_dim = embedding_dim
        self.pretrain = pretrain
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=(self.num_directions == 2),
        )
        if self.use_attention:
            assert window_size is not None, "window size must be set if use attention"
            self.attn = Attention(hidden_size * num_directions, window_size)
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion=nn.MSELoss(reduction='sum')
        self.prediction_layer = nn.Linear(
            self.hidden_size * self.num_directions, meta_data["vocab_size"]
        )

    def load_embedding(self, matrix):
        for i, row in enumerate(matrix):
            row_tensor = torch.tensor(row)  # 转换为张量
            self.embedder.embedding_layer.weight.data[i] = row_tensor  # 设置对应行的值

    def maml_train(self,input_dict,templates=None):
        return self.forward(input_dict,templates)

    def forward(self, input_dict,templates=None):
        if self.label_type == "anomaly":
            y = input_dict["window_anomalies"].long().view(-1)
        elif self.label_type == "next_log":
            y = input_dict["window_labels"].long().view(-1)
        self.batch_size = y.size()[0]
        x = input_dict["features"]
        x = self.embedder(x).to(self.device)
        # if self.feature_type == "semantics":
        #     if not self.use_tfidf:
        #         x = x.sum(dim=-2)  # add tf-idf
        outputs, _ = self.rnn(x.float())

        if self.use_attention:
            representation = self.attn(outputs)
        else:
            # representation = outputs.mean(dim=1)
            representation = outputs[:, -1, :]

        logits = self.prediction_layer(representation)
        y_pred = logits.softmax(dim=-1)
        # y=self.embedder(y,templates).to(self.device)
        y = input_dict["window_labels"].long().view(-1)
        loss = self.criterion(logits, y)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict
    
    def fit(self, train_loader, test_loader=None, epoches=10, learning_rate=1.0e-3,templates=None,print_result_path=None):
        self.to(self.device)
        logging.info(
            "Start training on {} batches with {}.".format(
                len(train_loader), self.device
            )
        )
        best_f1 = -float("inf")
        best_results = None
        worse_count = 0
        results=list()
        self.time_tracker["train"]=0.0
        for epoch in range(1, epoches + 1):
            epoch_time_start = time.time()
            model = self.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            batch_cnt = 0
            epoch_loss = 0
            tbar = tqdm(train_loader, desc="\r")
            num_batch = len(train_loader)
            for i, batch_input in enumerate(tbar):
                loss = self.maml_train(self.__input2device(batch_input),templates)["loss"]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
                batch_cnt += 1
                if not self.pretrain and test_loader is not None and (epoch % 1 == 0):
                # best_results=self.evaluate(test_loader)
                    eval_results = self.maml_evaluate(test_loader,templates,print_result_path)
                    if eval_results["f1s"] > best_f1:
                        best_f1 = eval_results["f1s"]
                        best_results = eval_results
                        best_results["epoches"] = int(epoch)
                        best_results["train_loss"] = loss
                        print(best_results)
                        sys.stdout.flush()
                        worse_count = 0
                    else:
                        worse_count +=1
                        if worse_count >= self.patience:
                            logging.info("Early stop at epoch: {}".format(epoch))
                            break
                #     results.append(best_results)
                tbar.set_description("Train loss: %.5f" % (epoch_loss / (i + 1)))
            epoch_loss = epoch_loss / batch_cnt
            epoch_time_elapsed = time.time() - epoch_time_start
            logging.info(
                "Epoch {}/{}, training loss: {:.5f} [{:.2f}s]".format(epoch, epoches, epoch_loss, epoch_time_elapsed)
            )
            self.time_tracker["train"] += epoch_time_elapsed

            # if not self.pretrain and test_loader is not None and (epoch % 1 == 0):
                # best_results=self.evaluate(test_loader)
            eval_results = self.maml_evaluate(test_loader,templates,print_result_path)
            best_results = eval_results
            best_results["epoches"] = int(epoch)

            if eval_results["f1s"] > best_f1:
                best_f1 = eval_results["f1s"]
                best_results = eval_results
                best_results["epoches"] = int(epoch)
                # best_results["train_loss"] = loss
                worse_count = 0
            else:
                worse_count +=1
                if worse_count >= self.patience:
                    logging.info("Early stop at epoch: {}".format(epoch))
                    break
            # print(best_results)
            # results.append(best_results)
        # return results
        return best_results


    def maml_evaluate(self,test_loader,templates=None,print_result_path=None):
        loss=0
        with torch.no_grad():
            y_pred = []
            store_dict = defaultdict(list)
            # templates_encoding=self.embedder.embedding_layer.encode(templates,show_progress_bar=False,convert_to_tensor=True,device=self.device)
            for batch_input in test_loader:
                # if templates is not None:
                #     batch_input["templates"]=torch.from_numpy(np.array(templates))
                # print(batch_input)
                return_dict = self.maml_train(self.__input2device(batch_input),templates)
                y_pred = return_dict["y_pred"]
                loss+=return_dict["loss"].item()
                y_prob_topk, y_pred_topk = torch.topk(y_pred, self.topk)  # b x topk

                store_dict["session_idx"].extend(
                    tensor2flatten_arr(batch_input["session_idx"])
                )
                store_dict["window_anomalies"].extend(
                    tensor2flatten_arr(batch_input["window_anomalies"])
                )
                store_dict["window_labels"].extend(
                    tensor2flatten_arr(batch_input["window_labels"])
                )
                store_dict["x"].extend(batch_input["features"].data.cpu().numpy())
                store_dict["y_pred_topk"].extend(y_pred_topk.data.cpu().numpy())
                store_dict["y_prob_topk"].extend(y_prob_topk.data.cpu().numpy())

            store_df = pd.DataFrame(store_dict)
            best_result = None
            # float("inf"):正无穷
            best_f1 = -float("inf")

            topkdf = pd.DataFrame(store_df["y_pred_topk"].tolist())
            hit_df = pd.DataFrame()
            for col in sorted(topkdf.columns):
                topk = col + 1
                hit = (topkdf[col] == store_df["window_labels"]).astype(int)
                hit_df[topk] = hit
                if col == 0:
                    acc_sum = 2 ** topk * hit
                else:
                    acc_sum += 2 ** topk * hit  # acc_num是一串数据series，长度为窗口总数（测试样本数），每个位数的数等于 2 ** k，k为该样本的y在pred中的索引
            # 将0处的（即pred的topk均没有y的测试样本）取很大的值，使得（转2）
            acc_sum[acc_sum == 0] = 2 ** (1 + len(topkdf.columns))
            hit_df["acc_num"] = acc_sum

            for col in sorted(topkdf.columns):
                topk = col + 1
                check_num = 2 ** topk
                # 2：使得 hit_df["acc_num"] > check_num 时，不是y在pred的索引在当前k（col）之后（前k个没有预测到y），就是整个topk都没有y，均为当前的异常情况
                # ~：0变1,1变0,~x == -x-1
                store_df["window_pred_anomaly_{}".format(topk)] = (
                    ~(hit_df["acc_num"] <= check_num)
                ).astype(int) # 若在k处命中y，k之前均为1（异常），k之后均为0（正常）
            # store_df.to_csv("store_{}_2.csv".format(dtype), index=False)


            if self.eval_type == "session":
                use_cols = ["session_idx", "window_anomalies"] + [
                    f"window_pred_anomaly_{topk}" for topk in range(1, self.topk + 1)
                ]
                # 分组后，sum将每一个session_idx的各列相加
                session_df = (
                    store_df[use_cols].groupby("session_idx", as_index=False).sum()
                )
            else:
                session_df = store_df
            # session_df.to_csv("session_{}_2.csv".format(dtype), index=False)

            for topk in range(1, self.topk + 1):
                pred = (session_df[f"window_pred_anomaly_{topk}"] > 0).astype(int)
                y = (session_df["window_anomalies"] > 0).astype(int)
                tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
                eval_results = {
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "TN": tn,
                    "recall": recall_score(y, pred),
                    "precision": precision_score(y, pred),
                    "f1s": f1_score(y, pred),
                    "acc":accuracy_score(y, pred),
                }
                if eval_results["f1s"] >= best_f1:
                    best_result = eval_results
                    best_f1 = eval_results["f1s"]
                    if print_result_path:
                        index_list=[i for i in range(len(pred)) if pred[i] == 1]
                        x_list=session_df["x"][index_list]
                        windows=[]
                        for x in x_list:
                            window=[]
                            for i in x:
                                window.append(templates[i])
                            windows.append(window)
                        windows=json.dumps(windows)
                        with open(print_result_path,"w") as f:
                            f.write(windows)
            best_result["loss"]=loss
            print(f'TP:{best_result["TP"]}, FP:{best_result["FP"]}, TN:{best_result["TN"]}, FN:{best_result["FN"]}')
            print(f'precision:{best_result["precision"]}, recall:{best_result["recall"]}, f1s:{best_result["f1s"]}')
            return best_result
        
    def __input2device(self, batch_input):
        return {k: v.to(self.device) for k, v in batch_input.items()}

class LogAnomaly(ForcastBasedModel_wos):
    def __init__(
        self,
        meta_data,
        hidden_size=100,
        num_directions=2,
        num_layers=4,
        window_size=None,
        use_attention=False,
        embedding_dim=32,
        model_save_path="./lstm_models",
        feature_type="sequentials",
        label_type="next_log",
        eval_type="window",
        topk=5,
        use_tfidf=False,
        freeze=False,
        gpu=-1,
        pretrain=True,
        **kwargs
    ):
        super().__init__(
            meta_data=meta_data,
            model_save_path=model_save_path,
            feature_type=feature_type,
            label_type=label_type,
            eval_type=eval_type,
            topk=topk,
            use_tfidf=use_tfidf,
            embedding_dim=embedding_dim,
            freeze=freeze,
            gpu=gpu,
        )
        num_labels = meta_data["num_labels"]
        self.vocab_size = meta_data["vocab_size"]
        self.feature_type = feature_type
        self.label_type = label_type
        self.hidden_size = hidden_size
        self.num_directions = num_directions
        self.window_size = window_size
        self.use_attention = use_attention
        self.use_tfidf = use_tfidf
        self.sbert_dim = 384
        self.embedding_dim = embedding_dim
        self.pretrain = pretrain
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=(self.num_directions == 2),
        )
        self.rnn2 = nn.LSTM(
            input_size=self.vocab_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=False,
        )
        if self.use_attention:
            assert window_size is not None, "window size must be set if use attention"
            self.attn = Attention(hidden_size * num_directions, window_size)
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion=nn.MSELoss(reduction='sum')
        self.prediction_layer = nn.Linear(
            self.hidden_size * self.num_directions + self.hidden_size, meta_data["vocab_size"]
        )

    def load_embedding(self, matrix):
        for i, row in enumerate(matrix):
            row_tensor = torch.tensor(row)  # 转换为张量
            self.embedder.embedding_layer.weight.data[i] = row_tensor  # 设置对应行的值

    def maml_train(self,input_dict,templates=None):
        return self.forward(input_dict,templates)

    def forward(self, input_dict,templates=None):
        if self.label_type == "anomaly":
            y = input_dict["window_anomalies"].long().view(-1)
        elif self.label_type == "next_log":
            y = input_dict["window_labels"].long().view(-1)
        self.batch_size = y.size()[0]
        vocab_count_tensor = torch.zeros(self.batch_size, self.vocab_size)
        x = input_dict["features"]
        for i in range(self.batch_size):
            window_counts = torch.bincount(x[i], minlength=self.vocab_size)
            vocab_count_tensor[i] = window_counts
        vocab_count_tensor = vocab_count_tensor.to(self.device)
        # print('size of x:', x.size())
        x = self.embedder(x).to(self.device)
        # if self.feature_type == "semantics":
        #     if not self.use_tfidf:
        #         x = x.sum(dim=-2)  # add tf-idf
        outputs, _ = self.rnn(x.float())
        output_2, _ = self.rnn2(vocab_count_tensor.float())

        if self.use_attention:
            representation = self.attn(outputs)
        else:
            # representation = outputs.mean(dim=1)
            representation = outputs[:, -1, :]

        representation_2 = output_2
        logits = self.prediction_layer(torch.cat((representation, representation_2), dim=1))
        y_pred = logits.softmax(dim=-1)
        # y=self.embedder(y,templates).to(self.device)
        y = input_dict["window_labels"].long().view(-1)
        loss = self.criterion(logits, y)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict
    
    def fit(self, train_loader, test_loader=None, epoches=10, learning_rate=1.0e-3,templates=None,print_result_path=None):
        self.to(self.device)
        logging.info(
            "Start training on {} batches with {}.".format(
                len(train_loader), self.device
            )
        )
        best_f1 = -float("inf")
        best_results = None
        worse_count = 0
        results=list()
        self.time_tracker["train"]=0.0
        for epoch in range(1, epoches + 1):
            epoch_time_start = time.time()
            model = self.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            batch_cnt = 0
            epoch_loss = 0
            tbar = tqdm(train_loader, desc="\r")
            num_batch = len(train_loader)
            for i, batch_input in enumerate(tbar):
                loss = self.maml_train(self.__input2device(batch_input),templates)["loss"]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
                batch_cnt += 1
                if not self.pretrain and test_loader is not None and (epoch % 1 == 0):
                # best_results=self.evaluate(test_loader)
                    eval_results = self.maml_evaluate(test_loader,templates,print_result_path)
                    # best_results = eval_results
                    # best_results["epoches"] = int(epoch)
                    # best_results["train_loss"] = loss
                    # print(best_results)
                    # sys.stdout.flush()
                    if eval_results["f1s"] > best_f1:
                        best_f1 = eval_results["f1s"]
                        best_results = eval_results
                        best_results["epoches"] = int(epoch)
                        best_results["train_loss"] = loss
                        print(best_results)
                        sys.stdout.flush()
                        worse_count = 0
                    else:
                        worse_count +=1
                        if worse_count >= self.patience:
                            logging.info("Early stop at epoch: {}".format(epoch))
                            break
                #     results.append(best_results)
                tbar.set_description("Train loss: %.5f" % (epoch_loss / (i + 1)))
            epoch_loss = epoch_loss / batch_cnt
            epoch_time_elapsed = time.time() - epoch_time_start
            logging.info(
                "Epoch {}/{}, training loss: {:.5f} [{:.2f}s]".format(epoch, epoches, epoch_loss, epoch_time_elapsed)
            )
            self.time_tracker["train"] += epoch_time_elapsed

            # if not self.pretrain and test_loader is not None and (epoch % 1 == 0):
                # best_results=self.evaluate(test_loader)
            eval_results = self.maml_evaluate(test_loader,templates,print_result_path)
            # best_results = eval_results
            # best_results["epoches"] = int(epoch)
            # print(best_results)
            # results.append(best_results)
            if eval_results["f1s"] > best_f1:
                best_f1 = eval_results["f1s"]
                best_results = eval_results
                best_results["epoches"] = int(epoch)
                # best_results["train_loss"] = loss
                # print(best_results)
                # sys.stdout.flush()
                worse_count = 0
            else:
                worse_count +=1
                if worse_count >= self.patience:
                    logging.info("Early stop at epoch: {}".format(epoch))
                    break
        # return results
        return best_results


    def maml_evaluate(self,test_loader,templates=None,print_result_path=None):
        loss=0
        with torch.no_grad():
            y_pred = []
            store_dict = defaultdict(list)
            # templates_encoding=self.embedder.embedding_layer.encode(templates,show_progress_bar=False,convert_to_tensor=True,device=self.device)
            for batch_input in test_loader:
                # if templates is not None:
                #     batch_input["templates"]=torch.from_numpy(np.array(templates))
                # print(batch_input)
                return_dict = self.maml_train(self.__input2device(batch_input),templates)
                y_pred = return_dict["y_pred"]
                loss+=return_dict["loss"].item()
                y_prob_topk, y_pred_topk = torch.topk(y_pred, self.topk)  # b x topk

                store_dict["session_idx"].extend(
                    tensor2flatten_arr(batch_input["session_idx"])
                )
                store_dict["window_anomalies"].extend(
                    tensor2flatten_arr(batch_input["window_anomalies"])
                )
                store_dict["window_labels"].extend(
                    tensor2flatten_arr(batch_input["window_labels"])
                )
                store_dict["x"].extend(batch_input["features"].data.cpu().numpy())
                store_dict["y_pred_topk"].extend(y_pred_topk.data.cpu().numpy())
                store_dict["y_prob_topk"].extend(y_prob_topk.data.cpu().numpy())

            store_df = pd.DataFrame(store_dict)
            best_result = None
            # float("inf"):正无穷
            best_f1 = -float("inf")

            topkdf = pd.DataFrame(store_df["y_pred_topk"].tolist())
            hit_df = pd.DataFrame()
            for col in sorted(topkdf.columns):
                topk = col + 1
                hit = (topkdf[col] == store_df["window_labels"]).astype(int)
                hit_df[topk] = hit
                if col == 0:
                    acc_sum = 2 ** topk * hit
                else:
                    acc_sum += 2 ** topk * hit  # acc_num是一串数据series，长度为窗口总数（测试样本数），每个位数的数等于 2 ** k，k为该样本的y在pred中的索引
            # 将0处的（即pred的topk均没有y的测试样本）取很大的值，使得（转2）
            acc_sum[acc_sum == 0] = 2 ** (1 + len(topkdf.columns))
            hit_df["acc_num"] = acc_sum

            for col in sorted(topkdf.columns):
                topk = col + 1
                check_num = 2 ** topk
                # 2：使得 hit_df["acc_num"] > check_num 时，不是y在pred的索引在当前k（col）之后（前k个没有预测到y），就是整个topk都没有y，均为当前的异常情况
                # ~：0变1,1变0,~x == -x-1
                store_df["window_pred_anomaly_{}".format(topk)] = (
                    ~(hit_df["acc_num"] <= check_num)
                ).astype(int) # 若在k处命中y，k之前均为1（异常），k之后均为0（正常）
            # store_df.to_csv("store_{}_2.csv".format(dtype), index=False)


            if self.eval_type == "session":
                use_cols = ["session_idx", "window_anomalies"] + [
                    f"window_pred_anomaly_{topk}" for topk in range(1, self.topk + 1)
                ]
                # 分组后，sum将每一个session_idx的各列相加
                session_df = (
                    store_df[use_cols].groupby("session_idx", as_index=False).sum()
                )
            else:
                session_df = store_df
            # session_df.to_csv("session_{}_2.csv".format(dtype), index=False)

            for topk in range(1, self.topk + 1):
                pred = (session_df[f"window_pred_anomaly_{topk}"] > 0).astype(int)
                y = (session_df["window_anomalies"] > 0).astype(int)
                tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
                eval_results = {
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "TN": tn,
                    "recall": recall_score(y, pred),
                    "precision": precision_score(y, pred),
                    "f1s": f1_score(y, pred),
                    "acc":accuracy_score(y, pred),
                }
                if eval_results["f1s"] >= best_f1:
                    best_result = eval_results
                    best_f1 = eval_results["f1s"]
                    if print_result_path:
                        index_list=[i for i in range(len(pred)) if pred[i] == 1]
                        x_list=session_df["x"][index_list]
                        windows=[]
                        for x in x_list:
                            window=[]
                            for i in x:
                                window.append(templates[i])
                            windows.append(window)
                        windows=json.dumps(windows)
                        with open(print_result_path,"w") as f:
                            f.write(windows)
            best_result["loss"]=loss
            print(f'TP:{best_result["TP"]}, FP:{best_result["FP"]}, TN:{best_result["TN"]}, FN:{best_result["FN"]}')
            print(f'precision:{best_result["precision"]}, recall:{best_result["recall"]}, f1s:{best_result["f1s"]}')
            return best_result
        
    def __input2device(self, batch_input):
        return {k: v.to(self.device) for k, v in batch_input.items()}