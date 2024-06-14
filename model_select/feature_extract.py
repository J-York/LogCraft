import time
import pandas as pd
import sys
from loglizer_core import dataloader, preprocessing
import json 
import numpy as np
import math
from loglizer_core.models import LogClustering, IsolationForest
from arch.unitroot import KPSS
from arch.utility.exceptions import InfeasibleTestException
from scipy.stats import skew, kurtosis, entropy
import os

save_path = r'metamatrix/Liberty.txt'

# ===============
# 进行 Landmark 特征的抽取
meta_feature_matrix = []

# 读取CSV文件
current_dir = os.getcwd()
folder_path = os.path.join(current_dir, 'data')
logfile_list = []
for root, dirs, files in os.walk(folder_path):
    for file_name in files:
        # 检查文件扩展名是否为 ".csv"
        if file_name.endswith('.log_structured.csv'):
            # 将 CSV 文件名添加到列表中
            csv_path = os.path.join(root, file_name)
            logfile_list.append(csv_path)
time_dict = dict()
for lf in logfile_list:
    print(f'precessing lf:{lf}')
    df = pd.read_csv(lf)
    print(df.head())
    start = time.time()
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
    # Isolation Forest

    model = IsolationForest()
    model.fit(x_train_ft)

    n_estimators = model.n_estimators

    estimators = model.estimators_

    leaf_node_counts = []
    nleaf_node_counts = []
    thresholds = []
    depths = []

    for i, estimator in enumerate(estimators):

        n_nodes = estimator.tree_.node_count

        depths.append(estimator.get_depth())

        split_features = estimator.tree_.feature
        split_thresholds = estimator.tree_.threshold
        thresholds.extend(split_thresholds)
        leaf_node_count = 0
        nleaf_node_count = 0
        for node_id in range(n_nodes):
            if split_features[node_id] != -2: 
                nleaf_node_count += 1
            else:
                leaf_node_count += 1
        leaf_node_counts.append(leaf_node_count)
        nleaf_node_counts.append(nleaf_node_count)

    landmark_features = []

    landmark_features.append(sum(leaf_node_counts)/len(leaf_node_counts))

    landmark_features.append(sum(nleaf_node_counts)/len(nleaf_node_counts))


    landmark_features.append(sum(thresholds)/len(thresholds))

    landmark_features.append(max(thresholds))

    # ==================================
    #  LogClustering 
    eid_windowed = np.array(eid_windowed)
    feature_extractor = preprocessing.FeatureExtractor()
    x_train_ft = feature_extractor.fit_transform(eid_windowed)
    max_dist = 0.3 # the threshold to stop the clustering process
    anomaly_threshold = 0.3 # the threshold for anomaly detection

    from loglizer_core.models import LogClustering
    model = LogClustering(max_dist=max_dist, anomaly_threshold=anomaly_threshold)
    model.fit(eid_windowed)

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
            avg_inter = model.compute_avg_inter_cluster_dist()
            min_inter = model.compute_min_inter_cluster_dist()
            return avg_inter, min_inter

    cluster_ana = LogClusterAnalyzer(model)
    cluster_landmarks = []

    # cluster size
    cluster_landmarks.append(len(model.cluster_size_dict))

    cluster_landmarks.append(max(model.cluster_size_dict.values())-min(model.cluster_size_dict.values()))

    cluster_landmarks.append(cluster_ana.get_inter_cluster_distances()[0])
    cluster_landmarks.append(cluster_ana.get_inter_cluster_distances()[1])

    landmark_features.extend(cluster_landmarks)

    middle = time.time()
    
    vector = []
    name = []

    def calculate_zero_ratio(arr):
        zero_count = np.count_nonzero(arr == 0) 
        total_elements = arr.size 
        zero_ratio = zero_count / total_elements 
        return zero_ratio

    def list_process(x, r_min=True, r_max=True, r_mean=True, r_std=True,
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

    def list_process_name(var):
        return [var + '_min', var + '_max', var + '_std', var + '_mean',
                var + '_skewness', var + '_kurtosis']

    def extracting_template_distribution(data):
        unique_nums = sorted(list(set(data.flatten())))
        result_array = np.zeros((len(unique_nums), len(data)))
        for i, row in enumerate(data):
            for num in row:
                index = unique_nums.index(num)
                result_array[index, i] += 1
        return result_array


    def frequency_analyze(array):
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
        vector.append(frequency_num)
        vector.append(frequency_max)
        vector.append(frequency_min)
        vector.append(frequency_var)
        vector.append(frequency_skew)
        vector.append(frequency_kurtosis)

        # Extend the `name` list with corresponding variable names
        name.extend(["template_num", "frequency_max", "frequency_min", "frequency_var", "frequency_skew", "frequency_kurtosis"])

    def template_statistical_characteristics(data):
        # basic
        array = extracting_template_distribution(data)
        row_max = np.nanmax(array,axis=1)
        row_min = np.nanmin(array,axis=1)
        row_std = np.nanstd(array,axis=1)
        vector.extend(list_process(row_max))
        name.extend(list_process_name('row_max'))
        vector.extend(list_process(row_min))
        name.extend(list_process_name('row_min'))
        vector.extend(list_process(row_std))
        name.extend(list_process_name('row_std'))
        # skew
        skewness_list = skew(array).reshape(-1, 1)
        skew_values = list_process(skewness_list)
        vector.extend(skew_values)
        name.extend(list_process_name('skewness'))

        # kurtosis
        kurtosis_list = kurtosis(array, axis=1)
        kurtosis_values = list_process(kurtosis_list)
        vector.extend(kurtosis_values)
        name.extend(list_process_name('kurtosis'))

        # 模板分布占比
        zero_count_list = []
        for row in array:
            zero_count_list.append(calculate_zero_ratio(row))
        zero_values = list_process(zero_count_list)
        vector.extend(zero_values)
        name.extend(list_process_name('zero'))

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
            

        vector.extend(list_process(statistic))
        name.extend(list_process_name('station_statistic'))
        vector.extend(list_process(pvalue))
        name.extend(list_process_name('station_pvalue'))
        vector.extend(list_process(lags))
        name.extend(list_process_name('station_lags'))
        vector.extend(list_process(is_station_5))
        name.extend(list_process_name('is_station_5'))

        # 熵
        # entropy_list = []
        # for row in array:
        #     entropy_item = entropy(row)
        #     entropy_list.append(entropy_item)

        # vector.extend(list_process(entropy_list))
        # name.extend(list_process_name('entropy'))

    def flatten_diagonally(x, diags=None):
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


    data = df["EID"].to_numpy()
    try:
        data = data.reshape((-1, 2000))
    except ValueError:
        data = data[:len(data) - (len(data) % 2000)]
        data = data.reshape((-1, 2000))

    template_statistical_characteristics(data)
    frequency_analyze(data)

    end = time.time()
    time_dict[lf] = (middle - start,end-middle,end-start)
    # 定义你希望删除的含义
    delete_list = ['frequency_max','frequency_min','row_max_min','row_min_min','row_min_max','row_min_std','row_min_mean','row_min_skewness','row_min_kurtosis','zero_min','station_statistic_min','station_pvalue_min','station_lags_min','is_station_5_min','is_station_5_max']

    # 遍历第二个列表，找出你不想要的含义，并从第一个列表中删除对应位置的数值
    for i in range(len(name)-1,-1,-1): #从后往前遍历，避免删除元素后索引变化
        if name[i] in delete_list: #如果第二个列表中的元素在你的删除列表中
            vector.pop(i) #从第一个列表中删除对应位置的数值
            name.pop(i) #从第二个列表中删除对应位置的含义

    static_features = vector
    meta_feature_vector = landmark_features + static_features
    meta_feature_matrix.append(meta_feature_vector)

    
np.savetxt(save_path, meta_feature_matrix)
with open("time_liberty.json",'w') as file:
    json.dump(time_dict,file)