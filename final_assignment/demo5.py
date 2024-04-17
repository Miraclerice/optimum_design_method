# -*- coding: utf-8 -*-
# @Author: MiracleRice
# Blog   : miraclerice.com

import numpy as np
import matplotlib.pyplot as plt
from utils import load_city_coord, load_China_coord, load_random_coord

"""
ACO(Ant Colony Optimization)
求解流程: 开始 --> 初始化参数 --> 构建解空间  --> 更新信息素 --> 得到最大迭代次数? --> 输出最优解 --> 结束
                                 |                               |
                                 <-- 当前迭代次数加1，清空路径记录表 <--
实例：TSP问题
"""
# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号 # python2在中文字符前加入u表示unicode编码,python3默认unicode编码
plt.rcParams['axes.unicode_minus'] = False

"""防止除0常数"""
EPS = 1e-4

######################################
# 初始化参数
######################################
# coord = load_city_coord('./city_coord.txt')
coord = load_China_coord('./China_coord.json')
# 城市个数
city_nums = len(coord)
# 坐标点间的距离
coord_dis = np.zeros((city_nums, city_nums))

# TODO: 优化，这里存储了两次，矩阵关于主对角线对称，有向图不能优化
for i in range(city_nums):
    for j in range(city_nums):
        if i != j:
            # 二维欧式距离
            coord_dis[i][j] = np.sqrt((coord[i][0] - coord[j][0]) ** 2 + (coord[i][1] - coord[j][1]) ** 2)
        else:
            # 注意除0
            coord_dis[i][j] = EPS

"""
ACO优化算法的参数
ant_nums:   蚂蚁的数量
Alpha:      信息启发因子(信息度重要程度参数)
Beta:       期望启发因子(启发式因子度重要程度参数)
Rho:        信息素蒸发系数
Q:          信息素增加强度
Eta:        启发式因子, 计算为距离的倒数
Tau:        信息素矩阵, 初始均为1
Tabu:       存储并记录路径的生成
Epoch:      最大迭代次数
NC:         迭代计数器
R_best:     记录最佳路径
L_best:     记录最优路径的长度
L_avg:      记录平均路径长度
"""
ant_nums = 50
Alpha = 1
Beta = 5
Rho = 0.1
Q = 100
Eta = 1 / coord_dis
Tau = np.ones((city_nums, city_nums))
Tabu = np.zeros((ant_nums, city_nums), dtype=np.int16)
Epoch = 200
NC = 1
R_best = np.zeros((Epoch, city_nums))
L_best = np.zeros(Epoch)
L_avg = np.zeros(Epoch)

######################################
# 构建解空间
######################################
"""迭代寻找最优解"""
while NC <= Epoch:
    # 随机产生各个蚂蚁的起点
    start = np.random.randint(0, city_nums, ant_nums, dtype=np.int16)
    Tabu[:, 0] = start
    # 定义一个城市索引，用于后续判断是否访问该城市
    coord_index = np.arange(city_nums)

    # 逐个蚂蚁路径选择
    for i in range(ant_nums):
        # 逐个地点路径选择
        for j in range(1, city_nums):
            # 已访问城市集合（禁忌表）
            tabu_vis = Tabu[i, :j]
            # 未访问城市集合，获取可访问城市的索引, 即后续可访问的城市
            # tabu_un = coord_index[~np.isin(coord_index, tabu_vis)]
            # 等同于上面语句
            tabu_un = np.setdiff1d(coord_index, tabu_vis)
            # 计算转移概率,不要使用zeros_like，计算为后概率为浮点数（tabu_un为索引值，定义为整形）
            prob = np.zeros(len(tabu_un))
            for k, city in enumerate(tabu_un):
                prob[k] = Tau[tabu_vis[-1], city] ** Alpha * Eta[tabu_vis[-1], city] ** Beta

            # 根据概率随机选择下一个城市(轮盘赌法选择)
            prob = prob / np.sum(prob)
            trans_prob_sum = np.cumsum(prob)
            trans_lst = np.where(trans_prob_sum >= np.random.rand())
            trans_index = tabu_un[trans_lst[0][0]]
            Tabu[i, j] = trans_index

    ######################################
    # 更新信息素
    ######################################
    # 计算各个蚂蚁的路径距离
    ants_dis = np.zeros(ant_nums)
    for i in range(ant_nums):
        path_mat = Tabu[i]
        for j in range(city_nums - 1):
            ants_dis[i] += coord_dis[path_mat[j], path_mat[j + 1]]
        ants_dis[i] += coord_dis[path_mat[-1], path_mat[0]]
    # 计算当前迭代次数的最短路径距离及平均距离，最后保留最优
    if NC == 1:
        L_best[NC - 1] = np.min(ants_dis)
        L_avg[NC - 1] = np.mean(ants_dis)
        R_best[NC - 1] = Tabu[np.argmin(ants_dis)]
    else:
        L_best[NC - 1] = min(L_best[NC - 2], np.min(ants_dis))
        L_avg[NC - 1] = np.mean(ants_dis)
        # 容易出错的步骤，永远取小迭代的最优路径
        R_best[NC - 1] = Tabu[np.argmin(ants_dis)] if L_best[NC - 1] == np.min(ants_dis) else \
            R_best[NC - 2]

    # 更新信息素
    pheromone = np.zeros((city_nums, city_nums))
    for i in range(ant_nums):
        # 获取路径记录的i处索引
        ts = Tabu[i]
        for j in range(city_nums - 1):
            pheromone[ts[j], ts[j + 1]] += Q / ants_dis[i]
        # 更新路径记录的最后一个索引到第一个索引的信息素
        pheromone[ts[-1], ts[0]] += Q / ants_dis[i]
    # 蚁周期模型(Ant-Cycle-Model)， 还有蚁密模型(Ant-density-Model)、蚁量模型(Ant-Quantity-Model)
    Tau = (1 - Rho) * Tau + pheromone

    ######################################
    # 判断是否终止
    ######################################
    # 下一轮迭代， 清空路径记录表
    NC += 1
    Tabu = np.zeros((ant_nums, city_nums), dtype=np.int16)

######################################
# 输出最优解
######################################
L_min, L_min_index = np.min(L_best), np.argmin(L_best)
R_min = R_best[L_min_index].astype(np.int16)
print('最短路径的迭代次数:', L_min_index)
print('最短距离为:', L_min)
print('最短路径为:', R_min)

# 绘图
plt.figure(1, dpi=300)
plt.grid(True)
for i in range(len(coord) - 1):
    plt.plot([coord[R_min[i], 0], coord[R_min[i + 1], 0]], [coord[R_min[i], 1], coord[R_min[i + 1], 1]], marker='o',
             markerfacecolor='none', c='deepskyblue')
plt.plot([coord[R_min[-1], 0], coord[R_min[0], 0]], [coord[R_min[-1], 1], coord[R_min[0], 1]], marker='o',
         markerfacecolor='none', c='deepskyblue')
for i in range(len(coord)):
    plt.text(coord[i, 0], coord[i, 1], f' {str(i + 1)}')
plt.text(coord[R_min[0], 0], coord[R_min[0], 1], '  起点', color='r', fontdict={'weight': 'bold', 'size': 12})
plt.text(coord[R_min[-1], 0], coord[R_min[-1], 1], '    终点', color='r', fontdict={'weight': 'bold', 'size': 12})
plt.xlabel('城市位置横坐标')
plt.ylabel('城市位置纵坐标')
plt.title('蚁群算法优化路径(最短距离:' + str(L_min) + ')')

plt.figure(2)
plt.plot(range(1, Epoch + 1), L_best, 'b', range(1, Epoch + 1), L_avg, 'r')
plt.legend(['最短距离', '平均距离'])
plt.xlabel('迭代次数')
plt.ylabel('距离')
plt.title('各代最短距离与平均距离对比')
plt.show()
