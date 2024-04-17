import numpy as np
from tqdm import tqdm  # 进度条设置
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib

matplotlib.use('TkAgg')
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


# ============蚁群算法求函数极值================

# =======适应度函数=====
def func(x, y):
    value = 20 * np.power(x * x - y * y, 2) - np.power(1 - y, 2) - 3 * np.power(1 + y, 2) + 0.3
    return value


# =======初始化参数====
m = 20  # 蚂蚁个数
G_max = 200  # 最大迭代次数
Rho = 0.9  # 信息素蒸发系数
P0 = 0.2  # 转移概率常数
XMAX = 5  # 搜索变量x最大值
XMIN = -5  # 搜索变量x最小值
YMAX = 5  # 搜索变量y最大值
YMIN = -5  # 搜索变量y最小值
X = np.zeros(shape=(m, 2))  # 蚁群 shape=(20, 2)
Tau = np.zeros(shape=(m,))  # 信息素
P = np.zeros(shape=(G_max, m))  # 状态转移矩阵
fitneess_value_list = []  # 迭代记录最优目标函数值
# ==随机设置蚂蚁初始位置==
for i in range(m):  # 遍历每一个蚂蚁
    X[i, 0] = np.random.uniform(XMIN, XMAX, 1)[0]  # 初始化x
    X[i, 1] = np.random.uniform(YMIN, YMAX, 1)[0]  # 初始化y
    Tau[i] = func(X[i, 0], X[i, 1])

step = 0.1;  # 局部搜索步长
for NC in range(G_max):  # 遍历每一代
    lamda = 1 / (NC + 1)
    BestIndex = np.argmin(Tau)  # 最优索引
    Tau_best = Tau[BestIndex]  # 最优信息素
    # ==计算状态转移概率===
    for i in range(m):  # 遍历每一个蚂蚁
        P[NC, i] = np.abs((Tau_best - Tau[i])) / np.abs(Tau_best) + 0.01  # 即例最优信息素的距离

    # =======位置更新==========
    for i in range(m):  # 遍历每一个蚂蚁
        # ===局部搜索====
        if P[NC, i] < P0:
            temp1 = X[i, 0] + (2 * np.random.random() - 1) * step * lamda  # x(2 * np.random.random() - 1) 转换到【-1,1】区间
            temp2 = X[i, 1] + (2 * np.random.random() - 1) * step * lamda  # y
        # ===全局搜索====
        else:
            temp1 = X[i, 0] + (XMAX - XMIN) * (np.random.random() - 0.5)
            temp2 = X[i, 0] + (YMAX - YMIN) * (np.random.random() - 0.5)

        # =====边界处理=====
        if temp1 < XMIN:
            temp1 = XMIN
        if temp1 > XMAX:
            temp1 = XMAX
        if temp2 < XMIN:
            temp2 = XMIN
        if temp2 > XMAX:
            temp2 = XMAX

        # ==判断蚂蚁是否移动(选更优===
        if func(temp1, temp2) < func(X[i, 0], X[i, 1]):
            X[i, 0] = temp1
            X[i, 1] = temp2

    # =====更新信息素========
    for i in range(m):  # 遍历每一个蚂蚁
        Tau[i] = (1 - Rho) * Tau[i] + func(X[i, 0], X[i, 1])  # (1 - Rho) * Tau[i] 信息蒸发后保留的
        index = np.argmin(Tau)  # 最小值索引
        value = Tau[index]  # 最小值
    fitneess_value_list.append(func(X[index, 0], X[index, 1]))  # 记录最优目标函数值

# ==打印结果===
min_index = np.argmin(Tau)  # 最优值索引
minX = X[min_index, 0]  # 最优变量x
minY = X[min_index, 1]  # 最优变量y
minValue = func(X[min_index, 0], X[min_index, 1])  # 最优目标函数值

print('最优变量x', minX, end='')
print('最优变量y', minY, end='\n')
print('最优目标函数值', minValue)

plt.plot(fitneess_value_list, label='迭代曲线')
plt.legend()
plt.show()