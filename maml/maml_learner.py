import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
import pandas as pd

from higherr.higher import innerloop_ctx
import os
# from    learner import Learner
from    copy import deepcopy
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from maml.maml_utils import set_device
from maml.maml_utils import plot_predict

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(
        self, 
        model,
        update_lr, 
        meta_lr,
        k_spt,
        k_qry,
        train_task_num,
        update_step,
        update_step_test,
        clip_val,
        gpu=-1,
        **kwargs,
        ):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = update_lr
        self.meta_lr = meta_lr
        self.k_spt = k_spt
        self.k_qry = k_qry
        self.task_num = train_task_num
        self.update_step = update_step
        self.update_step_test = update_step_test
        self.device=set_device(gpu)

        self.net = model
        self.clip_val=clip_val
        self.meta_optim = optim.SGD(self.net.parameters(), lr=self.meta_lr)   

    # def train(self, support,query):
    #     """
    #     :param x_spt:   [batch, setsz, seq_num, seq_len]
    #     :param y_spt:   [batch, setsz, seq_num]
    #     :param x_qry:   [batch, querysz, seq_num, seq_len]
    #     :param y_qry:   [batch, querysz, seq_num]
    #     :return:
    #     """
    #     # task_num, _, _, _ = x_spt.size()
    #     task_num=self.task_num
    #     qry_losses = []

    #     inner_opt = torch.optim.SGD(self.net.parameters(), lr=self.update_lr)
    #     # inner_opt= torch.optim.Adam(self.net.parameters(), lr=self.update_lr)
    #     # loss_fn = torch.nn.MSELoss()
    #     loss_fn = torch.nn.L1Loss()

    #     self.meta_optim.zero_grad()
    #     # 这里的task_num就是一个batch的task数量
    #     for i in range(task_num):
    #         # higher implementation
    #         with higher.innerloop_ctx(self.net, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
    #             # 1. run the i-th task and compute loss for k = 0 ~ self.update_steps
    #             for _ in range(self.update_step):
    #                 spt_loss = fnet(self.__input2device(support[i]))["loss"]
    #                 # spt_loss = loss_fn(y_pred_i, y_spt[i])
    #                 diffopt.step(spt_loss)

    #             # query_set meta backfoward
    #             qry_loss = fnet(self.__input2device(query[i]))["loss"]
    #             # qry_loss = loss_fn(y_pred_i_q, y_qry[i])
    #             qry_losses.append(qry_loss.detach())

    #             # update model's meta-parameters to optimize the query
    #             qry_loss.backward()

    #     self.meta_optim.step()
    #     qry_losses = sum(qry_losses) / task_num
    #     return qry_losses.item()

    def train(self, support,query,templates):
        """
        :param x_spt:   [batch, setsz, seq_num, seq_len]
        :param y_spt:   [batch, setsz, seq_num]
        :param x_qry:   [batch, querysz, seq_num, seq_len]
        :param y_qry:   [batch, querysz, seq_num]
        :return:
        """
        self.net.train()
        # task_num, _, _, _ = x_spt.size()
        task_num=self.task_num
        qry_losses = []

        inner_opt = torch.optim.SGD(self.net.parameters(), lr=self.update_lr)
        # inner_opt= torch.optim.Adam(self.net.parameters(), lr=self.update_lr)
        # loss_fn = torch.nn.MSELoss()
        # loss_fn = torch.nn.L1Loss()

        self.meta_optim.zero_grad()
        # 这里的task_num就是一个batch的task数量
        for i in range(task_num):
            # higher implementation
            with innerloop_ctx(self.net, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                # 1. run the i-th task and compute loss for k = 0 ~ self.update_steps
                for _ in range(self.update_step):
                    fnet.train()
                    spt_loss = fnet.maml_train(self.__input2device(support[i]),templates[i])["loss"]
                    diffopt.step(spt_loss)

                # query_set meta backfoward
                fnet.train()
                qry_loss = fnet.maml_train(self.__input2device(query[i]),templates[i])["loss"]
                # qry_loss = loss_fn(y_pred_i_q, y_qry[i])
                qry_losses.append(qry_loss.detach())

                # update model's meta-parameters to optimize the query
                qry_loss.backward()

        self.meta_optim.step()
        qry_losses = sum(qry_losses) / task_num
        return qry_losses.item()

    # def fineTunning(self, support,query, task_i, predict=False, pred_dir=None):
    #     """
    #     :param x_spt:   [setsz, seq_num, seq_len]
    #     :param y_spt:   [setsz, seq_num]
    #     :param x_qry:   [querysz, seq_num, seq_len]
    #     :param y_qry:   [querysz, seq_num]
    #     :return:
    #     """
    #     # assert len(x_spt.shape) == 3

    #     ft_net = deepcopy(self)
    #     loss_fn = torch.nn.L1Loss()
    #     optimizer_ft = torch.optim.SGD(ft_net.net.parameters(), lr=self.update_lr)
    #     test_loss = 0

    #     # non-higher implementation
    #     for _ in range(self.update_step_test):
    #         spt_loss = ft_net.net(self.__input2device(support))["loss"]
    #         # spt_loss = loss_fn(y_pred_spt, y_spt)

    #         optimizer_ft.zero_grad()
    #         spt_loss.backward()
    #         # clipping to avoid gradient explosion
    #         torch.nn.utils.clip_grad_norm_(ft_net.net.parameters(), self.clip_val)
    #         optimizer_ft.step()
            
    #     # query lossurn_dict
    #     return_dict=ft_net.net(self.__input2device(query))
    #     qry_loss = return_dict["loss"]
    #     # qry_loss = loss_fn(y_pred_qry, y_qry)
    #     test_loss = qry_loss.detach().item()

    #     # acc=self.evaluate(query,return_dict["y_pred"])["acc"]
    #     result=ft_net.evaluate(query,return_dict["y_pred"])
    #     # TODO:
    #     # acc=correct
        
    #     # prediction if pred is set to be True
    #     if predict == True:
    #         ts_pred, ts_ori = self.predictOneStep(ft_net, query)
    #         task_pred_dir = os.path.join(pred_dir, 'meta_test_task_{}'.format(task_i))
    #         if os.path.exists(task_pred_dir) is False:
    #             os.makedirs(task_pred_dir)

    #         for i in range(ts_pred.shape[0]):
    #             fig_name = os.path.join(task_pred_dir, 'query_{}.png'.format(i + 1))
    #             plot_predict(y_pred=ts_pred[i], y_true=ts_ori[i], fig_name=fig_name)
        
    #     return test_loss,result
    #     # return result
    def fineTunning(self, support,query,templates, task_i, predict=False, pred_dir=None):
        """
        :param x_spt:   [setsz, seq_num, seq_len]
        :param y_spt:   [setsz, seq_num]
        :param x_qry:   [querysz, seq_num, seq_len]
        :param y_qry:   [querysz, seq_num]
        :return:
        """
        # assert len(x_spt.shape) == 3

        # torch.save(self.net, "full_model.pt")
        # ft_net = torch.load("full_model.pt")
        ft_net = deepcopy(self.net)
        ft_net.train()
        optimizer_ft = torch.optim.SGD(ft_net.parameters(), lr=self.update_lr)

        # non-higher implementation
        # print(111111)
        for _ in range(self.update_step_test):
            spt_loss = ft_net.maml_train(self.__input2device(support),templates)["loss"]
            # spt_loss = loss_fn(y_pred_spt, y_spt)
            # print(111111)
            optimizer_ft.zero_grad()
            spt_loss.backward()
            # clipping to avoid gradient explosion
            torch.nn.utils.clip_grad_norm_(ft_net.parameters(), self.clip_val)
            optimizer_ft.step()
        # print(111111)
        # query lossurn_dict
        ft_net.eval()
        return_dict=ft_net.maml_evaluate([query],templates)
        
        return return_dict
        # return result
    
    def saveParams(self, save_path):
        torch.save(self.state_dict(), save_path)

    def predictOneStep(self, fnet, x, y):
        """
        :param x:           [setsz, seq_num, seq_len]
        :param y:           [setsz, seq_num]
        :return ts_pred:    [setsz, ts_len]
        :return ts_ori:     [setsz, ts_len]
        """
        assert len(x.shape) == 3 and len(y.shape) == 2
        setsz, _, _ = x.size()

        ts_pred = []
        ts_ori = []
        for i in range(setsz):
            ts_pred_i = fnet(x[i].unsqueeze(0))
            ts_pred_i_cpu = ts_pred_i.data.cpu().numpy()
            ts_pred_i_cpu = np.squeeze(ts_pred_i_cpu)
            ts_ori_i_cpu = y[i].data.cpu().numpy()
            ts_pred.append(ts_pred_i_cpu)
            ts_ori.append(ts_ori_i_cpu)
        
        return np.array(ts_pred), np.array(ts_ori)
    
    def __input2device(self, batch_input):
        return {k: v.to(self.device) for k, v in batch_input.items()}
    
    # def evaluate(self,input_dict,y_pred):
    #     # y_pred = []
    #     store_dict = defaultdict(list)
    #     # return_dict = self.net(input_dict)
    #     # y_pred = return_dict["y_pred"]
    #     # loss=return_dict["loss"]
    #     y_prob_topk, y_pred_topk = torch.topk(y_pred, self.net.topk)  # b x topk

    #     # store_dict["session_idx"]=tensor2flatten_arr(input_dict["session_idx"])
    #     # store_dict["window_anomalies"]=tensor2flatten_arr(input_dict["window_anomalies"])
    #     # store_dict["window_labels"]=tensor2flatten_arr(input_dict["window_labels"])
    #     # store_dict["x"]=input_dict["features"].data.cpu().numpy()
    #     # store_dict["y_pred_topk"]=y_pred_topk.data.cpu().numpy()
    #     # store_dict["y_prob_topk"]=y_prob_topk.data.cpu().numpy()
        
    #     store_dict["session_idx"].extend(
    #         tensor2flatten_arr(input_dict["session_idx"])
    #     )
    #     store_dict["window_anomalies"].extend(
    #         tensor2flatten_arr(input_dict["window_anomalies"])
    #     )
    #     store_dict["window_labels"].extend(
    #         tensor2flatten_arr(input_dict["window_labels"])
    #     )
    #     store_dict["x"].extend(input_dict["features"].data.cpu().numpy())
    #     store_dict["y_pred_topk"].extend(y_pred_topk.data.cpu().numpy())
    #     store_dict["y_prob_topk"].extend(y_prob_topk.data.cpu().numpy())
    #     store_df = pd.DataFrame(store_dict)
    #     best_result = None
    #     # float("inf"):正无穷
    #     best_f1 = -float("inf")
    #     topkdf = pd.DataFrame(store_df["y_pred_topk"].tolist())
    #     hit_df = pd.DataFrame()
    #     for col in sorted(topkdf.columns):
    #         topk = col + 1
    #         hit = (topkdf[col] == store_df["window_labels"]).astype(int)
    #         hit_df[topk] = hit
    #         if col == 0:
    #             acc_sum = 2 ** topk * hit
    #         else:
    #             acc_sum += 2 ** topk * hit  # acc_num是一串数据series，长度为窗口总数（测试样本数），每个位数的数等于 2 ** k，k为该样本的y在pred中的索引
    #     # 将0处的（即pred的topk均没有y的测试样本）取很大的值，使得（转2）
    #     acc_sum[acc_sum == 0] = 2 ** (1 + len(topkdf.columns))
    #     hit_df["acc_num"] = acc_sum

    #     for col in sorted(topkdf.columns):
    #         topk = col + 1
    #         check_num = 2 ** topk
    #         # 2：使得 hit_df["acc_num"] > check_num 时，不是y在pred的索引在当前k（col）之后（前k个没有预测到y），就是整个topk都没有y，均为当前的异常情况
    #         # ~：0变1,1变0,~x == -x-1
    #         store_df["window_pred_anomaly_{}".format(topk)] = (
    #             ~(hit_df["acc_num"] <= check_num)
    #         ).astype(int) # 若在k处命中y，k之前均为1（异常），k之后均为0（正常）
    #     # store_df.to_csv("store_{}_2.csv".format(dtype), index=False)


    #     if self.net.eval_type == "session":
    #         use_cols = ["session_idx", "window_anomalies"] + [
    #             f"window_pred_anomaly_{topk}" for topk in range(1, self.net.topk + 1)
    #         ]
    #         # 分组后，sum将每一个session_idx的各列相加
    #         session_df = (
    #             store_df[use_cols].groupby("session_idx", as_index=False).sum()
    #         )
    #     else:
    #         session_df = store_df
    #     # session_df.to_csv("session_{}_2.csv".format(dtype), index=False)
    #     correct=-1
    #     for topk in range(1, self.net.topk + 1):
    #         pred = (session_df[f"window_pred_anomaly_{topk}"] > 0).astype(int)
    #         y = (session_df["window_anomalies"] > 0).astype(int)
    #         eval_results = {
    #             "rc": recall_score(y, pred),
    #             "pc": precision_score(y, pred),
    #             "f1": f1_score(y, pred),
    #             "acc":accuracy_score(y, pred),
    #         }
    #         if eval_results["f1"] >= best_f1:
    #             correct=(pred==y).sum()
    #             best_result = eval_results
    #             best_f1 = eval_results["f1"]
    #     return best_result


    def test(self,test_loader,templates):
        self.net.eval()
        best_result=self.net.maml_evaluate(test_loader,templates)
        return best_result