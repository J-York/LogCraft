import numpy as np

array1 = np.load('/home/zhangshenglin/chenziang/MetaLogOD/MetaLogODNoValid/evaluation/total.npy')



array2 = np.genfromtxt(r'/home/zhangshenglin/chenziang/MetaLogOD/MetaLogODNoValid/evaluation/data.csv', delimiter=',')


combined_array = np.concatenate((array1, array2), axis=1)

# 打印合并后的数组形状
print('数组的大小：', combined_array.shape)

np.save('statistic.npy', combined_array)
exit(0)

X_meta = np.loadtxt("../feature/feature_matrix_model_based.txt")
print('数组的大小：', X_meta.shape)
exit(0)
combined_array = np.concatenate((array1, X_meta), axis=1)

# 打印合并后的数组形状
print('数组的大小：', combined_array.shape)
np.savetxt('../feature/feature_matrix_model_based.txt', combined_array)