import time
from structure import MetaODClass
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
import numpy as np
import pandas as pd
import warnings
import sys
args = sys.argv

warnings.filterwarnings("ignore", category=RuntimeWarning) # 暂时忽略冷启动导致的runtime error

id2dataset = {0:"BGL",1:"HDFS",2:"Liberty",3:"Spirit",4:"ThunderBird"}
index = eval(args[1])
######################################读取数据并归一化###########################################

# 读取评估矩阵
X = np.load("./evaluation/total_4dataset.npy")
X_train = np.concatenate((X[:index], X[index+1:]), axis=0)
X_test = X[index]

print("Select Test set is " + id2dataset[index])
print("Evaluation matrix shape:" + str(X_train.shape))

# 读取特征矩阵并按列归一化（防止某一特征影响过大）到[0,1]
X_meta = np.loadtxt("./feature/feature_matrix_59.txt")
# X_meta_first_36 = X_meta[:, :31]  # 前31项
# X_meta_last_4 = X_meta[:, -4:]   # 最后4项
# 使用 numpy.concatenate() 来拼接数组
# X_meta = np.concatenate((X_meta_first_36, X_meta_last_4), axis=1)
# X_meta = X_meta[:, :36] + X_meta[:, -4:]
X_train_meta = np.concatenate((X_meta[:index], X_meta[index+1:]), axis=0)
X_test_meta = X_meta[index]
X_train_meta = (X_train_meta - np.nanmin(X_train_meta,axis=0)) / (np.nanmax(X_train_meta,axis=0) - np.nanmin(X_train_meta,axis=0))
X_train_meta = np.nan_to_num(X_train_meta)
# X_train_meta = X_train_meta[:9, :]
print("Feature matrix shape:" + str(X_train_meta.shape))



# # 测试集按行最大最小归一化，便于最终的比较
# normal_test = (X_test - np.nanmin(X_test,axis=1)[:, np.newaxis]) / (np.nanmax(X_test,axis=1)[:, np.newaxis] - np.nanmin(X_test,axis=1)[:, np.newaxis])
# normal_test = np.nan_to_num(normal_test)
# print("Normalized test set:" + str(normal_test))


######################################训练并进行预测############################################
# 训练集加入训练
EMF = MetaODClass(X_train, learning_method='sgd') 
start = time.time()
EMF.train(n_iter=200, meta_features=X_train_meta, min_rate=0.05, max_rate=0.2)

# import pickle
# with open("model.pkl",'wb') as file:
#     pickle.dump(EMF, file)

U = EMF.user_vectors
V = EMF.item_vectors

pred_scores = np.dot(U, V.T) # 训练集上收敛的NDCG的LOSS

print('Ultimate NDCG loss on the training set:', ndcg_score(X_train, pred_scores))
end = time.time() - start
print("The time spent during the training process is:" + str(end))

X_test_meta = np.reshape(X_test_meta, (1, -1))
predicted_scores = EMF.predict(X_test_meta)


######################################判断预测结果准确性###########################################

sorted_test_indexes = np.argsort(-X_test) # 测试集正确排序（从高到小）
sorted_predict_indexes = np.argsort(-predicted_scores[0]) # 实际排序
print("Test set sort index:" + str(sorted_test_indexes))
print("Predict sort index:" + str(sorted_predict_indexes))

normal_test_2d = np.reshape(X_test, (1, -1)) # 转化为二维才能计算NDCG
predicted_scores_2d = np.reshape(predicted_scores[0], (1, -1))

dist1 = ndcg_score(normal_test_2d, predicted_scores_2d,k=3)
print("Test set NDCG score:" + str(dist1))

# 预测得分归一化
predicted_scores = (predicted_scores - np.nanmin(predicted_scores,axis=1)[:, np.newaxis]) / (np.nanmax(predicted_scores,axis=1)[:, np.newaxis] - np.nanmin(predicted_scores,axis=1)[:, np.newaxis])
predicted_scores = np.nan_to_num(predicted_scores)
print("The normalized predict score is:" + str(predicted_scores))
