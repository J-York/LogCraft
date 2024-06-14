import os
import pandas as pd
import numpy as np
from model_select.loglizer_core import dataloader, preprocessing
from model_select.loglizer_core.models import LogClustering, IsolationForest
from arch.unitroot import KPSS
from arch.utility.exceptions import InfeasibleTestException
from scipy.stats import skew, kurtosis, entropy
from model_select.structure import MetaODClass
from sklearn.metrics import ndcg_score
import time
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import json


class LogClusterAnalyzer:

    def __init__(self, model):
        self.model = model

    def get_cluster_sizes(self):
        sizes = []
        for i in range(len(self.model.representatives)):
            sizes.append(self.model.cluster_size_dict.get(i, 0))
        return sizes

    def get_cluster_centers(self):
        return np.array(self.model.representatives)


    def get_inter_cluster_distances(self):
        avg_inter = self.model.compute_avg_inter_cluster_dist()
        min_inter = self.model.compute_min_inter_cluster_dist()
        return avg_inter, min_inter


class Analysis():
    def __init__(self,vector,name) -> None:
        self.vector=vector
        self.name=name

    def calculate_zero_ratio(self,arr):
        zero_count = np.count_nonzero(arr == 0)  # 统计零元素的个数
        total_elements = arr.size  # 数组总元素个数
        zero_ratio = zero_count / total_elements  # 计算零元素占比
        return zero_ratio

    def list_process(self,x, r_min=True, r_max=True, r_mean=True, r_std=True,
                     r_skew=True, r_kurtosis=True):
        x = np.asarray(x).reshape(-1, 1)
        return_list = []
        if r_min:
            return_list.append(np.nanmin(x))

        if r_max:
            return_list.append(np.nanmax(x))

        if r_std:
            return_list.append(np.nanstd(x))
        if r_mean:
            return_list.append(np.nanmean(x))
        if r_skew:
            return_list.append(skew(x, nan_policy='omit')[0])
        if r_kurtosis:
            return_list.append(kurtosis(x, nan_policy='omit')[0])

        return return_list

    def list_process_name(self,var):
        return [var + '_min', var + '_max', var + '_std', var + '_mean',
                var + '_skewness', var + '_kurtosis']

    def extracting_template_distribution(self,data):
        # 计算所有不同数字的有序列表
        unique_nums = sorted(list(set(data.flatten())))
        # 创建空的二维数组
        result_array = np.zeros((len(unique_nums), len(data)))
        # 遍历原始矩阵的每一行
        for i, row in enumerate(data):
            # 对每一个数字，找到它在有序列表中的索引
            for num in row:
                index = unique_nums.index(num)
                # 在结果数组中对应的位置加一
                result_array[index, i] += 1
        # 返回结果数组
        return result_array


    def frequency_analyze(self,array):
        """
        Calculate the frequency of a numpy array.

        Args:
            array (numpy.ndarray): Input array.

        Returns:
            None. The calculated frequencies are stored in the `vector` list.

        """
        # Flatten the array
        data = array.flatten()

        # Calculate the frequency of each value
        counts = np.bincount(data)

        # Calculate frequency statistics
        frequency_num = len(counts)
        frequency_max = np.max(counts)
        frequency_min = np.min(counts)
        frequency_var = np.var(counts)
        frequency_skew = skew(counts)
        frequency_kurtosis = kurtosis(counts)

        # Store the frequency statistics in the `vector` list
        self.vector.append(frequency_num)
        self.vector.append(frequency_max)
        self.vector.append(frequency_min)
        self.vector.append(frequency_var)
        self.vector.append(frequency_skew)
        self.vector.append(frequency_kurtosis)

        # Extend the `name` list with corresponding variable names
        self.name.extend(["template_num", "frequency_max", "frequency_min", "frequency_var", "frequency_skew", "frequency_kurtosis"])

    def template_statistical_characteristics(self,data):
        # basic
        array = self.extracting_template_distribution(data)
        row_max = np.nanmax(array,axis=1)
        row_min = np.nanmin(array,axis=1)
        row_std = np.nanstd(array,axis=1)
        self.vector.extend(self.list_process(row_max))
        self.name.extend(self.list_process_name('row_max'))
        self.vector.extend(self.list_process(row_min))
        self.name.extend(self.list_process_name('row_min'))
        self.vector.extend(self.list_process(row_std))
        self.name.extend(self.list_process_name('row_std'))
        # skew
        skewness_list = skew(array).reshape(-1, 1)
        skew_values = self.list_process(skewness_list)
        self.vector.extend(skew_values)
        self.name.extend(self.list_process_name('skewness'))

        # kurtosis
        kurtosis_list = kurtosis(array, axis=1)
        kurtosis_values = self.list_process(kurtosis_list)
        self.vector.extend(kurtosis_values)
        self.name.extend(self.list_process_name('kurtosis'))

        # 模板分布占比
        zero_count_list = []
        for row in array:
            zero_count_list.append(self.calculate_zero_ratio(row))
        zero_values = self.list_process(zero_count_list)
        self.vector.extend(zero_values)
        self.name.extend(self.list_process_name('zero'))

        # 平稳度分析
        statistic = []
        pvalue = []
        lags = []
        is_station_5 = []
        for index, row in enumerate(array):
            try:
                kpss = KPSS(row)
                statistic.append(kpss.stat)
                pvalue.append(kpss.pvalue)
                lags.append(kpss.lags)
                is_station_5.append((kpss.critical_values['5%'] <= kpss.stat).astype(int))
            except InfeasibleTestException:
                print("第" + str(index) + "行无法计算平稳度")
                print(row)
                continue
            

        self.vector.extend(self.list_process(statistic))
        self.name.extend(self.list_process_name('station_statistic'))
        self.vector.extend(self.list_process(pvalue))
        self.name.extend(self.list_process_name('station_pvalue'))
        self.vector.extend(self.list_process(lags))
        self.name.extend(self.list_process_name('station_lags'))
        self.vector.extend(self.list_process(is_station_5))
        self.name.extend(self.list_process_name('is_station_5'))

        # 熵
        # entropy_list = []
        # for row in array:
        #     entropy_item = entropy(row)
        #     entropy_list.append(entropy_item)

        # self.vector.extend(list_process(entropy_list))
        # self.name.extend(list_process_name('entropy'))

    def flatten_diagonally(self,x, diags=None):
        diags = np.array(diags)
        if x.shape[1] > x.shape[0]:
            diags += x.shape[1] - x.shape[0]
        n = max(x.shape)
        ndiags = 2 * n - 1
        i, j = np.indices(x.shape)
        d = np.array([])
        for ndi in range(ndiags):
            if diags != None:
                if not ndi in diags:
                    continue
            d = np.concatenate((d, x[i == j + (n - 1) - ndi]))
        return d


def feature_extract(log_name):
    data_dir = os.path.join("../data_preprocessed/", log_name)
    log_file = f"{log_name}.log_structured.csv"
    df=pd.read_csv(os.path.join(data_dir, log_file), engine="c", na_filter=False, memory_map=True)
    # 将EventTemplate列转换为整数类型
    df["EID"] = pd.factorize(df["EventId"])[0]
    # df.to_csv('/home/zhangshenglin/jiyuhe/NaviProject/data/BGL/result_10w/BGL_10w.log_structured.csv', index=False)
    eid_list = df['EID'].tolist()

    eid_windowed = []
    window = []
    for i in range(len(eid_list)):
        window.append(eid_list[i])
        if (i+1) % 20 == 0:
            eid_windowed.append(window)
            window = []
    if len(window) != 0:
        # 最后一组元素个数     
        last_group_len = len(window)  
        # 需要补齐的0元素个数
        zero_num = 20 - last_group_len  
        # 补齐0
        window.extend([0] * zero_num)
        eid_windowed.append(window)

    eid_windowed = np.array(eid_windowed)
    x_train = eid_windowed
    feature_extractor = preprocessing.FeatureExtractor()
    x_train_ft = feature_extractor.fit_transform(eid_windowed)

    # ==================================
    # 使用 Isolation Forest 提取日志数据集的特征

    model = IsolationForest()
    model.fit(x_train_ft)

    n_estimators = model.n_estimators
    print(f"决策树数量: {n_estimators}")

    # 获取森林中的每个决策树
    estimators = model.estimators_

    leaf_node_counts = []
    nleaf_node_counts = []
    thresholds = []
    depths = []

    # 遍历每个决策树并提取结构信息
    for i, estimator in enumerate(estimators):
    #     print(f"决策树 {i + 1} 的结构信息:")

        # 获取决策树的节点数量
        n_nodes = estimator.tree_.node_count
    #     print(f"节点数量: {n_nodes}")

        # 获取决策树的深度
        depths.append(estimator.get_depth())

        # 获取每个节点的分裂特征和阈值
        split_features = estimator.tree_.feature
        split_thresholds = estimator.tree_.threshold
        thresholds.extend(split_thresholds)
        leaf_node_count = 0
        nleaf_node_count = 0
        for node_id in range(n_nodes):
            if split_features[node_id] != -2:  # -2表示叶子节点
    #             print(f"节点 {node_id}: 分裂特征 = {split_features[node_id]}, 阈值 = {split_thresholds[node_id]}")
                nleaf_node_count += 1
            else:
    #             print(f"节点 {node_id}: 叶子节点")
                leaf_node_count += 1
        leaf_node_counts.append(leaf_node_count)
        nleaf_node_counts.append(nleaf_node_count)
    #     print('叶子节点数：{}， 非叶子节点数：{}'.format(leaf_node_count, nleaf_node_count))

    # 决策树数量 平均叶子节点数量 平均非叶子节点数 平均深度 最高深度 平均分裂阈值 最高分裂阈值 最低分裂阈值
    landmark_features = []

    #决策树数量
    # landmark_features.append(n_estimators)

    # 平均叶子节点数量
    landmark_features.append(sum(leaf_node_counts)/len(leaf_node_counts))

    # 平均非叶子节点数
    landmark_features.append(sum(nleaf_node_counts)/len(nleaf_node_counts))

    # 平均深度
    # landmark_features.append(sum(depths)/len(depths))

    # 最高深度
    # landmark_features.append(max(depths))

    # 平均分裂阈值
    landmark_features.append(sum(thresholds)/len(thresholds))

    # 最高分裂阈值
    landmark_features.append(max(thresholds))

    # 最低分裂阈值
    # landmark_features.append(min(thresholds))

    # ====================================
    # 使用 LogClustering 进行原特征的抽取

    eid_windowed = np.array(eid_windowed)
    feature_extractor = preprocessing.FeatureExtractor()
    x_train_ft = feature_extractor.fit_transform(eid_windowed)
    max_dist = 0.3 # the threshold to stop the clustering process
    anomaly_threshold = 0.3 # the threshold for anomaly detection
    model = LogClustering(max_dist=max_dist, anomaly_threshold=anomaly_threshold)
    model.fit(eid_windowed)
    cluster_ana = LogClusterAnalyzer(model)
    cluster_landmarks = []

    # cluster size
    cluster_landmarks.append(len(model.cluster_size_dict))

    # min cluster size 
    # cluster_landmarks.append(min(model.cluster_size_dict.values()))

    # cluster size 极差
    cluster_landmarks.append(max(model.cluster_size_dict.values())-min(model.cluster_size_dict.values()))

    # 平均类间距 最小类间距
    cluster_landmarks.append(cluster_ana.get_inter_cluster_distances()[0])
    cluster_landmarks.append(cluster_ana.get_inter_cluster_distances()[1])

    landmark_features.extend(cluster_landmarks)

    
    vector = []
    name = []
    data = df["EID"].to_numpy()
    try:
        data = data.reshape((-1, 2000))
    except ValueError:
        data = data[:len(data) - (len(data) % 2000)]
        data = data.reshape((-1, 2000))
    analysis=Analysis(vector,name)
    analysis.template_statistical_characteristics(data)
    analysis.frequency_analyze(data)

    # 定义你希望删除的含义
    delete_list = ['frequency_max','frequency_min','row_max_min','row_min_min','row_min_max','row_min_std','row_min_mean','row_min_skewness','row_min_kurtosis','zero_min','station_statistic_min','station_pvalue_min','station_lags_min','is_station_5_min','is_station_5_max']

    # 遍历第二个列表，找出你不想要的含义，并从第一个列表中删除对应位置的数值
    for i in range(len(name)-1,-1,-1): #从后往前遍历，避免删除元素后索引变化
        if name[i] in delete_list: #如果第二个列表中的元素在你的删除列表中
            vector.pop(i) #从第一个列表中删除对应位置的数值
            name.pop(i) #从第二个列表中删除对应位置的含义

    static_features = vector
    meta_feature_vector = landmark_features + static_features
    meta_feature_vector=np.array(meta_feature_vector)
    return meta_feature_vector

def model_select(log_name):
    X_test_meta=feature_extract(log_name)
    # 读取评估矩阵
    id2dataset = {0:"BGL",1:"HDFS",2:"Liberty",3:"Spirit",4:"ThunderBird"}
    X_train = np.load("../model_select/evaluation/total_4dataset.npy")
    print("Evaluation matrix shape:" + str(X_train.shape))
    # 读取特征矩阵并按列归一化（防止某一特征影响过大）到[0,1]
    X_train_meta = np.loadtxt("../model_select/feature/feature_matrix_59.txt")
    X_train_meta = (X_train_meta - np.nanmin(X_train_meta,axis=0)) / (np.nanmax(X_train_meta,axis=0) - np.nanmin(X_train_meta,axis=0))
    X_train_meta = np.nan_to_num(X_train_meta)
    print("Feature matrix shape:" + str(X_train_meta.shape))
    ######################################训练并进行预测############################################
    # 训练集加入训练
    EMF = MetaODClass(X_train, learning_method='sgd')
    start = time.time()
    EMF.train(n_iter=200, meta_features=X_train_meta, min_rate=0.05, max_rate=0.2)
    U = EMF.user_vectors
    V = EMF.item_vectors

    pred_scores = np.dot(U, V.T) # 训练集上收敛的NDCG的LOSS

    print('Ultimate NDCG loss on the training set:', ndcg_score(X_train, pred_scores))
    end = time.time() - start
    print("The time spent during the training process is:" + str(end))
    print(X_test_meta.shape)
    X_test_meta = np.reshape(X_test_meta, (1, -1))
    # X_test_meta=np.nan_to_num(X_test_meta)
    # print(X_test_meta.isnull().sum())
    predicted_scores = EMF.predict(X_test_meta)


    ######################################判断预测结果准确性###########################################

    # sorted_test_indexes = np.argsort(-X_test) # 测试集正确排序（从高到小）
    sorted_predict_indexes = np.argsort(-predicted_scores[0]) # 实际排序
    # print("Test set sort index:" + str(sorted_test_indexes))
    print("Predict sort index:" + str(sorted_predict_indexes))

    # normal_test_2d = np.reshape(X_test, (1, -1)) # 转化为二维才能计算NDCG
    # predicted_scores_2d = np.reshape(predicted_scores[0], (1, -1))

    # dist1 = ndcg_score(normal_test_2d, predicted_scores_2d,k=3)
    # print("Test set NDCG score:" + str(dist1))

    # 预测得分归一化
    predicted_scores = (predicted_scores - np.nanmin(predicted_scores,axis=1)[:, np.newaxis]) / (np.nanmax(predicted_scores,axis=1)[:, np.newaxis] - np.nanmin(predicted_scores,axis=1)[:, np.newaxis])
    predicted_scores = np.nan_to_num(predicted_scores)
    print("The normalized predict score is:" + str(predicted_scores))
    with open("../model_select/evaluation/total.json",'r') as f:
        data = json.load(f)
    top1_model=data['models'][sorted_predict_indexes[0]][0]
    return top1_model