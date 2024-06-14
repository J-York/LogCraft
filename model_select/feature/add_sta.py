import os
import numpy as np
import pandas as pd

data = np.loadtxt("feature_matrix_59.txt")
data = data.tolist()
def read_csv_and_extract_f1s(file_path):
    df = pd.read_csv(file_path)
    f1s_column = df['f1s'].tolist()
    return f1s_column

# 定义文件夹路径
folder_path = '/home/zhangshenglin/chenziang/MetaLogOD/MetaLogODNoValid/evaluation/sat_results'

index = 0
# 遍历文件夹中的CSV文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        f1s_column = read_csv_and_extract_f1s(file_path)
        for item in f1s_column:
            data[index].append(item)
    index += 1
data = np.array(data)
np.savetxt("feature_matrix_model_based.txt",data)