import numpy as np

# data = np.load("total.npy")
data = [[],[],[],[]]
import os
import pandas as pd

# 定义函数来读取CSV文件并提取"F1S"列
def read_csv_and_extract_f1s(file_path):
    df = pd.read_csv(file_path)
    # df = df[df['Unnamed: 0'].str.contains('\+') == False]
    f1s_column = df['f1s'].tolist()
    return f1s_column

# 定义文件夹路径
folder_path = '/home/zhangshenglin/chenziang/data/sat_results_new1'

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
# 打印合并后的二维数组
# print(data)
# 指定要保存的文件路径和文件名
file_path = 'data.csv'

# 使用savetxt函数将数组写入CSV文件
np.savetxt(file_path, data, delimiter=',',fmt='%.4f')
# np.save('statistic.npy', data)