import numpy as np
import pandas as pd

# 从npy文件中加载数据
data = np.load('total_4dataset.npy')

# 将数据转换为DataFrame
df = pd.DataFrame(data)
df = df.round(4)
# 将DataFrame保存为CSV文件
df.to_csv('data.csv', index=False)
