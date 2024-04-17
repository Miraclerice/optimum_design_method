# -*- coding: utf-8 -*-
# @Author: MiracleRice
# Blog   : miraclerice.com

import numpy as np
from functools import reduce

from matplotlib import pyplot as plt


class Ant(object):
    def __init__(self, nCity, graph, pheromone):
        self.nCity = nCity  # 城市数
        self.graph = graph  # 城市地图
        self.pheromone = pheromone  # 信息素地图
        self.cityEnabled = [True] * nCity  # 尚未到达的城市标记
        self.nMove = 0  # 移动步数
        self.dTotal = 0.0  # 已经走过的距离
        self.initData()  # 初始化出生点

    # 随机选择城市
    def randCity(self):
        return np.random.randint(0, self.nCity - 1)

    # 初始化
    def initData(self):
        self.nowCity = self.randCity()  # 随机选择一个城市
        self.path = [self.nowCity]  # 保存当前走过的城市
        self.cityEnabled[self.nowCity] = False  # 当前城市不再探索
        self.nMove = 1  # 初始时的移动计数

    # 计算到达第i个城市的概率
    def getOneProb(self, i):
        ALPHA = 1.0
        BETA = 2.0
        dis = self.graph[self.nowCity][i]
        phe = self.pheromone[self.nowCity][i]
        # 计算移动到该城市的概率
        if not self.cityEnabled[i]:
            return 0
        else:
            return pow(phe, ALPHA) * pow(1 / dis, BETA)

    def choiceNext(self):
        # 前往所有城市的概率
        pSlct = [self.getOneProb(i) for i in range(self.nCity)]
        pSum = np.cumsum(pSlct)
        # 生成一个随机数，这个随机数在pSum中落入的区间就是选择的城市
        pTemp = np.random.uniform(0.0, pSum[-1])
        return np.searchsorted(pSum, pTemp)

    # 移动到新的城市
    def moveTo(self, city):
        self.path.append(city)  # 添加目标城市
        self.cityEnabled[city] = False  # 目标城市不可再探索
        # 总路程增加当前城市到目标城市的距离
        self.dTotal += self.graph[self.nowCity][city]
        self.nowCity = city  # 更新当前城市
        self.nMove += 1  # 移动次数

    def run(self):
        self.initData()
        while self.nMove < self.nCity:
            next_city = self.choiceNext()
            self.moveTo(next_city)
        self.dTotal += self.graph[self.path[0]][self.path[-1]]
        return self.dTotal


# 更新信息素
RHO = 0.5  # 信息素挥发系数
Q = 50  # 信息素增加强度系数


def updatePheromone(nCity, pheromone, ants):
    # 初始化蚂蚁在两两城市间的信息素, 50行50列
    temp = np.zeros([nCity, nCity])
    # 遍历每只蚂蚁对象
    for ant in ants:
        for i in range(1, nCity):  # 遍历该蚂蚁经过的每个城市
            st, ed = ant.path[i - 1], ant.path[i]
            # 在两个城市间留下信息素，浓度与总距离成反比
            temp[st, ed] += Q / ant.dTotal
            temp[ed, st] = temp[st, ed]  # 信息素矩阵轴对称
    return pheromone * RHO + temp


import copy


# xs, ys为城市的x和y坐标
# nAnts为蚂蚁个数, nIter为迭代次数
def aco(xs, ys, nAnts, nIter):
    nCity = len(xs)
    xMat, yMat = xs - xs.reshape(-1, 1), ys - ys.reshape(-1, 1)
    graph = np.sqrt(xMat ** 2 + yMat ** 2)
    pheromone = np.ones([nCity, nCity])  # 信息素矩阵
    ants = [Ant(nCity, graph, pheromone) for _ in range(nAnts)]
    best = Ant(nCity, graph, pheromone)  # 初始化最优解
    best.dTotal = np.inf
    bestAnts = []  # 输出并保存
    for i in range(nIter):
        for ant in ants:
            ant.pheromone = pheromone
            ant.run()
            # 与当前最优蚂蚁比较步行的总距离
            if ant.dTotal < best.dTotal:
                # 更新最优解
                best = copy.deepcopy(ant)
        print(f"{i},{best.dTotal}")
        # 更新信息素
        pheromone = updatePheromone(nCity, pheromone, ants)
        bestAnts.append(best)
    return bestAnts


# 每个城市的x和y坐标
xs = np.array([
    178, 272, 176, 171, 650, 499, 267, 703, 408, 437, 491, 74, 532,
    416, 626, 42, 271, 359, 163, 508, 229, 576, 147, 560, 35, 714,
    757, 517, 64, 314, 675, 690, 391, 628, 87, 240, 705, 699, 258,
    428, 614, 36, 360, 482, 666, 597, 209, 201, 492, 294])
ys = np.array([
    170, 395, 198, 151, 242, 556, 57, 401, 305, 421, 267, 105, 525,
    381, 244, 330, 395, 169, 141, 380, 153, 442, 528, 329, 232, 48,
    498, 265, 343, 120, 165, 50, 433, 63, 491, 275, 348, 222, 288,
    490, 213, 524, 244, 114, 104, 552, 70, 425, 227, 331])
xMat, yMat = xs - xs.reshape(-1, 1), ys - ys.reshape(-1, 1)
distance_graph = np.sqrt(xMat ** 2 + yMat ** 2)

if __name__ == '__main__':
    bAnts = aco(xs, ys, 50, 300)
    index = bAnts[-1].path
    index = index + [index[0]]
    plt.plot(xs[index], ys[index], marker='*')
    plt.tight_layout()
    plt.show()
