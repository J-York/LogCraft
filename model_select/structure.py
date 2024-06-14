import numpy as np
from sklearn.metrics import ndcg_score
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor



def sigmoid(x, a=1):
    return 1 / (1 + np.exp(-1 * a * x))


def sigmoid_derivate(x, a=1):
    return sigmoid(x, a) * (1 - sigmoid(x, a))

class MetaODClass(object):
    def __init__(self,
                 train_performance,
                 learning_method):
        self.ratings = train_performance
        print("training evaluation performance shape:" + str(train_performance.shape))
        self.n_samples, self.n_models = train_performance.shape # n_samples:数据集个数 n_models:模型个数
        self.learning_method = learning_method # 留作扩展
        self.train_loss_ = [0] # 保存每一次训练的training loss
        self.learning_rates_ = [] # 学习率数组
        self.scalar_ = None

    def train(self, meta_features, n_iter=10, max_depth=10, n_estimators=100, 
              max_rate=1.05, min_rate=0.1, n_steps=10):
        """ Train model for n_iter iterations from scratch."""
        
        # 元特征标准化
        # StandardScaler计算每个元特征的均值和标准差
        self.scalar_ = StandardScaler().fit(meta_features)
        # 对元特征进行标准化
        meta_features_scaled = self.scalar_.transform(meta_features)

        self.user_vectors = meta_features_scaled # U矩阵初始化

        self.n_factors = self.user_vectors.shape[1] # n_factors: 元特征个数

        self.item_vectors = np.random.normal(scale=1. / self.n_factors,
                                          size=(self.n_models, self.n_factors)) # V矩阵初始化
        

        # CLR 维护周期性学习率数组 
        lr_list = np.linspace(min_rate, max_rate, n_steps)
        lr_list_reverse = lr_list[::-1]
        learning_rate_full = np.concatenate([np.concatenate((lr_list, lr_list_reverse[1:-2]))]*n_iter)

        # learning_rate_full = []
        # for i in range(n_iter):
        #     learning_rate_full = learning_rate_full + [min_rate,max_rate]
        # if max_rate >= min_rate:
        #     max_rate = max_rate / 2

        # learning_rate_full = np.concatenate(([min_rate], [max_rate]) * n_iter)
        self.learning_rate_ = min_rate
        self.learning_rates_.append(self.learning_rate_)

        current_iter = 1
        non_progress_counter  = 1 # 总计数器，记录没有明显优化的次数
        while current_iter <= n_iter:
            # 选择当前迭代的学习率
            self.learning_rate_ = learning_rate_full[current_iter - 1]
            self.learning_rates_.append(self.learning_rate_)

            # 确保非空
            self.user_vectors[np.isnan(self.user_vectors)] = 0

            # 回归函数
            self.regr_multirf = MultiOutputRegressor(RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1))

            self.regr_multirf.fit(meta_features_scaled, self.user_vectors)

            # 确定当前迭代的LOSS 使用训练集的LOSS
            ndcg_s = []
            for w in range(self.ratings.shape[0]):
                ndcg_s.append(ndcg_score([self.ratings[w, :]], [
                    np.dot(self.user_vectors[w, :], self.item_vectors.T)])) 
            self.train_loss_.append(np.mean(ndcg_s)) # 计算每一个数据集评估矩阵和UV转置的ndcg_score，取均值

            print('Meta Learnint epoch', current_iter, 'training loss',
                  self.train_loss_[-1],'learning rate', self.learning_rates_[-1])
            
            # 当测试集的loss不再显著变化时跳出循环
            tolerance = 0.0001
            if len(self.train_loss_) >= 2:
                # 防止数组越界
                relative_change = abs((self.train_loss_[-1] - self.train_loss_[-2]) / self.train_loss_[-2]) # 相对变化，判断测试集新老loss的变化比率
                if relative_change <= tolerance:
                    non_progress_counter += 1
                else:
                    non_progress_counter = 1

            if non_progress_counter > 5:
                break

            # 随机打乱

            train_indices = list(range(self.n_samples))  # 所有训练数据集的索引列表
            np.random.shuffle(train_indices)  # 随机打乱索引列表

            for h in train_indices:
                uh = self.user_vectors[h, :].reshape(1, -1)  # 当前数据集的user_vectors
                grads = []  # 存储梯度值的列表

                for i in range(self.n_models):
                    vi = self.item_vectors[i, :].reshape(-1, 1)  # 当前模型的item_vectors
                    phis = []  # 存储临时值phi的列表
                    rights = []  # 存储临时值right的列表
                    rights_v = []  # 存储临时值right_v的列表

                    # 移除当前模型的索引
                    other_models = list(range(self.n_models))
                    other_models.remove(i)

                    for j in other_models:
                        vj = self.item_vectors[j, :].reshape(-1, 1)  # 其他模型的item_vectors
                        temp_vt = sigmoid(np.ndarray.item(np.matmul(uh, (vj - vi))), a=1)
                        temp_vt_derivative = sigmoid_derivate(np.ndarray.item(np.matmul(uh, (vj - vi))), a=1)
                        phis.append(temp_vt)
                        rights.append(temp_vt_derivative * (vj - vi))
                        rights_v.append(temp_vt_derivative * uh)
                    
                    phi = np.sum(phis) + 1.5
                    rights = np.asarray(rights).reshape(len(other_models), self.n_factors)
                    rights_v = np.asarray(rights_v).reshape(len(other_models), self.n_factors)

                    right = np.sum(np.asarray(rights), axis=0)
                    right_v = np.sum(np.asarray(rights_v), axis=0) 

                    grad = (10 ** (self.ratings[h, i]) - 1) / (phi * (np.log(phi)) ** 2) * right  # 计算模型梯度值
                    grad_v = (10 ** (self.ratings[h, i]) - 1) / (phi * (np.log(phi)) ** 2) * right_v  # 计算item_vectors梯度值

                    self.item_vectors[i, :] += self.learning_rate_ * grad_v  # 更新当前模型的item_vectors

                    grads.append(grad)

                grads_uh = np.asarray(grads)
                grad_uh = np.sum(grads_uh, axis=0)  # 计算user_vectors的梯度值

                self.user_vectors[h, :] -= self.learning_rate_ * grad_uh  # 更新当前数据集的user_vectors

            current_iter += 1

        # disable unnecessary information
        self.ratings = None
        return self

    def remove_nan_columns(self,test_meta):
        nan_positions = np.isnan(test_meta)

        if nan_positions.any():
            nan_indices = np.where(nan_positions)[1]  # 获取 NaN 的索引列表，选择列索引

            # 删除 NaN 值
            test_meta = np.delete(test_meta, nan_indices, axis=1)

            # 删除对应列
            self.item_vectors = np.delete(self.item_vectors, nan_indices, axis=1)

            return test_meta

    def fill_nan_columns(self, test_meta):
        nan_positions = np.isnan(test_meta)

        if nan_positions.any():
            nan_indices = np.where(nan_positions)[1]  # 获取 NaN 的索引列表，选择列索引

            # 将 NaN 值填充为 0
            for column_index in np.unique(nan_indices):
                test_meta[:, column_index] = 0

            # 更新 item_vectors 中对应列的值
            for column_index in np.unique(nan_indices):
                self.item_vectors[:, column_index] = 0

            return test_meta


    def check_and_prepare_meta(self, test_meta):
        # 检查 test_meta 是否有 NaN 并进行处理
        try:
            test_meta = check_array(test_meta)
        except ValueError:
            # 有 NaN 处理
            test_meta = self.fill_nan_columns(test_meta)
        return test_meta

    def predict(self, test_meta):
        self.item_vectors_old = self.item_vectors.copy()
        test_meta = self.check_and_prepare_meta(test_meta)
        test_meta_scaled = self.scalar_.transform(test_meta)
        test_meta_scaled = self.regr_multirf.predict(test_meta_scaled) # 回归函数映射，得到Utest

        predicted_scores = np.dot(test_meta_scaled, self.item_vectors.T) # 得到Ptest
        self.item_vectors = self.item_vectors_old
        assert (predicted_scores.shape[0] == test_meta.shape[0])
        assert (predicted_scores.shape[1] == self.n_models)

        return predicted_scores
    
