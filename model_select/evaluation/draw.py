import numpy as np
import matplotlib.pyplot as plt
X = np.load("/home/zhangshenglin/chenziang/MetaLogOD/MetaLogODNoValid/evaluation/total.npy")

X_test = X[3]
def plot_bar(data):
    # 创建一个新的图形
    plt.figure(figsize=(8, 6))
    
    # 绘制柱状图
    plt.bar(range(len(data)), data)
    
    # 添加标题和标签
    plt.title('Bar Plot')
    plt.xlabel('X Label')
    plt.ylabel('Y Label')
    
    # 显示图形
    plt.savefig("figure.png")


# 调用函数绘制柱状图
plot_bar(X_test)