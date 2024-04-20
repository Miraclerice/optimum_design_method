# -*- coding: utf-8 -*-
# @Author: MiracleRice
# Blog   : miraclerice.com
import os.path
import time
import numpy as np
from utils import load_city_coord, load_China_coord, load_random_coord, Logger
import matplotlib.pyplot as plt

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号 # python2在中文字符前加入u表示unicode编码,python3默认unicode编码
plt.rcParams['axes.unicode_minus'] = False
"""防止除0常数"""
EPS = 1e-4


class Ant(object):
    """
    人工蚂蚁
    :parameter
    city_nums: 城市个数
    coord_dis: 城市间的距离
    alpha: 信息素重要程度
    beta: 启发式因子
    rho: 信息素蒸发系数
    Q: 信息素增加强度
    eta: 启发式因子
    tau: 信息素矩阵
    tabu: 记录路径
    """

    def __init__(self, city_nums, coord_dis, tau, alpha, beta):
        self.city_nums = city_nums
        self.coord_dis = coord_dis
        self.alpha = alpha
        self.beta = beta
        self.eta = 1 / coord_dis
        self.tau = tau
        self.tabu = np.zeros(city_nums, dtype=np.int16)

    def move(self):
        # 随机初始化起点
        start = np.random.randint(0, self.city_nums, dtype=np.int16)
        self.tabu[0] = start
        # 定义一个城市索引，用于后续判断是否访问该城市
        coord_index = np.arange(self.city_nums)
        # 逐个地点路径选择
        for i in range(1, self.city_nums):
            # 已访问城市集合（禁忌表）
            tabu_vis = self.tabu[:i]
            tabu_un = np.setdiff1d(coord_index, tabu_vis)
            # 计算转移概率
            prob = np.zeros(len(tabu_un))
            for j, city in enumerate(tabu_un):
                prob[j] = self.tau[tabu_vis[-1], city] ** self.alpha * self.eta[tabu_vis[-1], city] ** self.beta

            # 根据概率随机选择下一个城市（轮盘赌法）
            prob = prob / np.sum(prob)
            prob_sum = np.cumsum(prob)
            # np.where返回的是元祖
            next_city_idx = tabu_un[np.where(prob_sum >= np.random.rand())[0][0]]
            self.tabu[i] = next_city_idx
        return self.tabu, self.calculate_dis(self.tabu)

    def calculate_dis(self, route):
        """计算蚂蚁遍历所有城市后距离（回到起点）"""
        dis = 0
        for i in range(self.city_nums - 1):
            dis += self.coord_dis[route[i], route[i + 1]]
        dis += self.coord_dis[route[-1], route[0]]
        return dis


class TSP(object):
    """
    TSP问题
    """

    def __init__(self, coord, ants_nums=50, alpha=1, beta=5, rho=0.1, Q=100, epoch=200):
        self.coord = coord
        self.ants_nums = ants_nums
        self.city_nums = len(coord)
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.epoch = epoch
        self.tau = np.ones((self.city_nums, self.city_nums))
        self.tabu = np.zeros((self.ants_nums, self.city_nums), dtype=np.int16)
        self.route_bs = np.zeros((self.epoch, self.city_nums))
        self.len_bs = np.zeros(self.epoch)
        self.len_avg = np.zeros(self.epoch)
        self.coord_dis = self.get_coord_dis()

    def train(self, mode=1):
        """
        mode:
            1: 基本AS(蚁周期模型, ant-cycle-model)
            2: 基本AS(蚁量模型, ant-quantity-model)
            3: 基本AS(蚁密模型, ant-density-model)
            4: EAS(精英蚂蚁系统)
            5: RBAS(基于排序的蚂蚁系统)
            6: SAAS(自适应蚂蚁系统)
            7: MMAS(最大最小蚂蚁系统) TODO: 待实现，这个需要根据信息素修改路径，需要验证下一个城市索引是否合理
        """
        start_time = time.time()
        for i in range(self.epoch):
            ants_dis = np.zeros(self.ants_nums)
            # 按照旅行距离排序，短的前w-1位蚂蚁
            w = int(self.ants_nums * 0.8)
            rank = np.zeros(w - 1)
            k = 0
            for j in range(self.ants_nums):
                ant = Ant(self.city_nums, self.coord_dis, self.tau, self.alpha, self.beta)
                tabu, dis = ant.move()
                self.tabu[j] = tabu
                ants_dis[j] = dis
                if j == 0:
                    rank[k] = dis
                else:
                    if dis < rank[k - 1]:
                        rank[k] = dis
                        k += 1
            # 将索引值按照长度排序，小的在前
            rank_w = np.argsort(rank)

            if mode == 6:
                # 自适应蚂蚁系统, 这里t为一次迭代
                rho_min = 0.1
                if self.rho >= rho_min:
                    self.rho = 0.95 * self.rho

            self.update_dis(i, ants_dis)
            self.update_tau(i, ants_dis, mode, rank_w)
            # 清空路径记录表，这里其实没有必要
            self.tabu = np.zeros((self.ants_nums, self.city_nums), dtype=np.int16)
        end_time = time.time()
        consume_time = end_time - start_time
        # 输出最优解
        model = ['AS(蚁周期模型)', 'AS(蚁量模型)', 'AS(蚁密模型)', '精英蚂蚁系统', '基于排序的蚂蚁系统',
                 '自适应蚂蚁系统', '最大最小蚂蚁系统']
        res = self.result_save(model[mode - 1], consume_time)
        self.draw(res, model[mode - 1])

    def update_tau(self, nc, ants_dis, mode, rank):
        """更新信息素矩阵"""
        pheromone = np.zeros((self.city_nums, self.city_nums))
        coord_dis = self.coord_dis
        Q = self.Q
        # 精英蚂蚁系统的权重系数, 一般设置为城市数量大小
        e = self.city_nums
        # 已知最优路径长度和距离，由于更新了直接就是nc
        len_bs_konwd = self.len_bs[nc]
        route_bs_konwd = self.route_bs[nc]
        if mode == 4:
            # EAS
            for i in range(self.ants_nums):
                route = self.tabu[i]
                # 查看是否出现精英蚂蚁
                if np.all(route == route_bs_konwd):
                    for j in range(self.city_nums - 1):
                        pheromone[route[j], route[j + 1]] += (Q / ants_dis[i] + e / len_bs_konwd)
                    pheromone[route[-1], route[0]] += (Q / ants_dis[i] + e / len_bs_konwd)
                else:
                    for j in range(self.city_nums - 1):
                        pheromone[route[j], route[j + 1]] += Q / ants_dis[i]
                    pheromone[route[-1], route[0]] += Q / ants_dis[i]
        elif mode == 5 or mode == 6:
            # RBAS
            w = len(rank) + 1
            for rank, i in enumerate(rank):
                route = self.tabu[i]
                # 查看是否出现精英蚂蚁
                if np.all(route == route_bs_konwd):
                    for j in range(self.city_nums - 1):
                        pheromone[route[j], route[j + 1]] += ((w - rank - 1) / self.len_bs[nc] + w / len_bs_konwd)
                else:
                    for j in range(self.city_nums - 1):
                        pheromone[route[j], route[j + 1]] += ((w - rank - 1) / self.len_bs[nc])
        else:
            for i in range(self.ants_nums):
                route = self.tabu[i]
                if mode == 1:
                    # ant-cycle-model
                    for j in range(self.city_nums - 1):
                        pheromone[route[j], route[j + 1]] += Q / ants_dis[i]
                    pheromone[route[-1], route[0]] += Q / ants_dis[i]
                elif mode == 2:
                    # ant-quantity-model
                    for j in range(self.city_nums - 1):
                        pheromone[route[j], route[j + 1]] += Q / coord_dis[route[j], route[j + 1]]
                    pheromone[route[-1], route[0]] += Q / coord_dis[route[-1], route[0]]
                elif mode == 3:
                    # ant-density-model
                    for j in range(self.city_nums - 1):
                        pheromone[route[j], route[j + 1]] += Q
                    pheromone[route[-1], route[0]] += Q

        self.tau = (1 - self.rho) * self.tau + pheromone

    def update_dis(self, nc, ants_dis):
        """更新某一次迭代的最短距离，平均距离"""
        if nc == 0:
            self.len_bs[nc] = np.min(ants_dis)
            self.len_avg[nc] = np.mean(ants_dis)
            self.route_bs[nc] = self.tabu[np.argmin(ants_dis)]
        else:
            self.len_bs[nc] = min(self.len_bs[nc - 1], np.min(ants_dis))
            self.len_avg[nc] = np.mean(ants_dis)
            self.route_bs[nc] = self.tabu[np.argmin(ants_dis)] if self.len_bs[nc] == np.min(ants_dis) else \
                self.route_bs[nc - 1]

    def get_coord_dis(self):
        n = self.city_nums
        coord_dis = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    # 二维欧式距离
                    coord_dis[i][j] = np.sqrt((coord[i][0] - coord[j][0]) ** 2 + (coord[i][1] - coord[j][1]) ** 2)
                else:
                    # 注意除0
                    coord_dis[i][j] = EPS
        return coord_dis

    def result_save(self, mode, consume_time):
        """输出结果到控制台和指定日志文件"""
        logger = Logger('./log', 'ACO.log')
        len_min, len_min_idx = np.min(self.len_bs), np.argmin(self.len_bs)
        route_min = self.route_bs[len_min_idx].astype(np.int16)
        logger.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}\n")
        logger.write(f'当前模式：{mode} \n')
        logger.write(f'总迭代次数: {self.epoch}\t最短路径的迭代次数: {len_min_idx} \n')
        logger.write(f'最短路径的长度: {len_min:.4f} \n')
        logger.write(f'最短路径的路径: {route_min} \n')
        logger.write(f'运行耗时: {consume_time:.2f}s\n')
        return len_min, len_min_idx, route_min

    def draw(self, res, mode):
        len_min, len_min_idx, route_min = res
        coord = self.coord
        plt.figure(1)
        plt.grid(True)
        for i in range(len(coord) - 1):
            plt.plot([coord[route_min[i]][0], coord[route_min[i + 1]][0]],
                     [coord[route_min[i]][1], coord[route_min[i + 1]][1]], marker='o',
                     markerfacecolor='none', c='deepskyblue')
        plt.plot([coord[route_min[-1], 0], coord[route_min[0], 0]],
                 [coord[route_min[-1], 1], coord[route_min[0], 1]], marker='o', markerfacecolor='none', c='deepskyblue')
        for i in range(len(coord)):
            plt.text(coord[i, 0], coord[i, 1], f' {str(i)}')
        plt.text(coord[route_min[0], 0], coord[route_min[0], 1], '  起点', color='r',
                 fontdict={'weight': 'bold', 'size': 12})
        plt.text(coord[route_min[-1], 0], coord[route_min[-1], 1], '    终点', color='r',
                 fontdict={'weight': 'bold', 'size': 12})
        plt.xlabel('城市位置横坐标')
        plt.ylabel('城市位置纵坐标')
        plt.title(f'{mode}(最短距离:{len_min:.4f})')
        # assert os.path.exists('./img_rand'), '文件夹不存在, 请先创建文件夹'
        img_path = f'./image/img_{self.city_nums}_{self.epoch}'
        if not os.path.exists(img_path):
            os.makedirs(img_path)

        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        plt.savefig(f'{img_path}/{mode}1.png', bbox_inches='tight')

        plt.figure(2)
        plt.plot(range(1, self.epoch + 1), self.len_bs, 'b', range(1, self.epoch + 1), self.len_avg, 'r')
        plt.legend(['最短距离', '平均距离'])
        plt.xlabel('迭代次数')
        plt.ylabel('距离')
        plt.title(mode + '各代最短距离与平均距离对比')
        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        plt.savefig(f'{img_path}/{mode}2.png', bbox_inches='tight')
        plt.show()

    def clear(self):
        """将重要参数重新初始化"""
        self.tau = np.ones((self.city_nums, self.city_nums))
        self.tabu = np.zeros((self.ants_nums, self.city_nums), dtype=np.int16)
        self.route_bs = np.zeros((self.epoch, self.city_nums))
        self.len_bs = np.zeros(self.epoch)
        self.len_avg = np.zeros(self.epoch)


if __name__ == '__main__':
    # coord = load_China_coord('./China_coord.json')
    coord = load_China_coord('./China_admin_center_coord.json')
    # coord = load_random_coord(100)
    ants_nums = 50
    # tsp = TSP(coord, ants_nums, rho=0.5)
    tsp = TSP(coord, ants_nums, epoch=200)
    # tsp.train(mode=1)
    # tsp.clear()
    #
    # tsp.train(mode=2)
    # tsp.clear()
    #
    # tsp.train(mode=3)
    # tsp.clear()
    #
    # tsp.train(mode=4)
    # tsp.clear()
    #
    # tsp.train(mode=5)
    # tsp.clear()
    # print('模式5已完成')
    # print(tsp.rho)
    #
    # print('模式6：自适应蚂蚁系统')
    tsp.rho = 1
    tsp.train(mode=6)
    # tsp.clear()
