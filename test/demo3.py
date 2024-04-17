import copy
import sys
import random
import math
from functools import reduce

from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsEllipseItem, QGraphicsTextItem, \
    QGraphicsLineItem, QPushButton, QVBoxLayout, QWidget, QGraphicsView
from PyQt5.QtCore import Qt, QTimer

(ALPHA, BETA, RHO, Q) = (1.0, 2.0, 0.5, 100.0)
# 城市数，蚁群
(city_num, ant_num) = (50, 50)
distance_x = [
    178, 272, 176, 171, 650, 499, 267, 703, 408, 437, 491, 74, 532,
    416, 626, 42, 271, 359, 163, 508, 229, 576, 147, 560, 35, 714,
    757, 517, 64, 314, 675, 690, 391, 628, 87, 240, 705, 699, 258,
    428, 614, 36, 360, 482, 666, 597, 209, 201, 492, 294]
distance_y = [
    170, 395, 198, 151, 242, 556, 57, 401, 305, 421, 267, 105, 525,
    381, 244, 330, 395, 169, 141, 380, 153, 442, 528, 329, 232, 48,
    498, 265, 343, 120, 165, 50, 433, 63, 491, 275, 348, 222, 288,
    490, 213, 524, 244, 114, 104, 552, 70, 425, 227, 331]
# 城市距离和信息素
distance_graph = [[0.0 for col in range(city_num)] for raw in range(city_num)]
pheromone_graph = [[1.0 for col in range(city_num)] for raw in range(city_num)]


# ----------- 蚂蚁 -----------
class Ant:
    # 初始化
    def __init__(self, ID):
        self.ID = ID  # ID
        self.__clean_data()  # 随机初始化出生点

    # 初始数据
    def __clean_data(self):
        self.path = []  # 当前蚂蚁的路径
        self.total_distance = 0.0  # 当前路径的总距离
        self.move_count = 0  # 移动次数
        self.current_city = -1  # 当前停留的城市
        self.open_table_city = [True for i in range(city_num)]  # 探索城市的状态
        city_index = random.randint(0, city_num - 1)  # 随机初始出生点
        self.current_city = city_index
        self.path.append(city_index)
        self.open_table_city[city_index] = False
        self.move_count = 1

    # 选择下一个城市
    def __choice_next_city(self):
        next_city = -1
        select_citys_prob = [0.0 for i in range(city_num)]  # 存储去下个城市的概率
        total_prob = 0.0
        # 获取去下一个城市的概率
        for i in range(city_num):
            if self.open_table_city[i]:
                try:
                    # 计算概率：与信息素浓度成正比，与距离成反比
                    select_citys_prob[i] = pow(pheromone_graph[self.current_city][i], ALPHA) * pow(
                        (1.0 / distance_graph[self.current_city][i]), BETA)
                    total_prob += select_citys_prob[i]
                except ZeroDivisionError as e:
                    print('Ant ID: {ID}, current city: {current}, target city: {target}'.format(ID=self.ID,
                                                                                                current=self.current_city,
                                                                                                target=i))
                    sys.exit(1)
        # 轮盘选择城市
        if total_prob > 0.0:
            # 产生一个随机概率,0.0-total_prob
            temp_prob = random.uniform(0.0, total_prob)
            for i in range(city_num):
                if self.open_table_city[i]:
                    # 轮次相减
                    temp_prob -= select_citys_prob[i]
                    if temp_prob < 0.0:
                        next_city = i
                        break
        # 未从概率产生，顺序选择一个未访问城市
        # if next_city == -1:
        #     for i in range(city_num):
        #         if self.open_table_city[i]:
        #             next_city = i
        #             break
        if (next_city == -1):
            next_city = random.randint(0, city_num - 1)
            while ((self.open_table_city[next_city]) == False):  # if==False,说明已经遍历过了
                next_city = random.randint(0, city_num - 1)
        # 返回下一个城市序号
        return next_city

    # 计算路径总距离
    def __cal_total_distance(self):
        temp_distance = 0.0
        for i in range(1, city_num):
            start, end = self.path[i], self.path[i - 1]
            temp_distance += distance_graph[start][end]
        # 回路
        end = self.path[0]
        temp_distance += distance_graph[start][end]
        self.total_distance = temp_distance

    # 移动操作
    def __move(self, next_city):
        self.path.append(next_city)
        self.open_table_city[next_city] = False
        self.total_distance += distance_graph[self.current_city][next_city]
        self.current_city = next_city
        self.move_count += 1

    # 搜索路径
    def search_path(self):
        # 初始化数据
        self.__clean_data()
        # 搜素路径，遍历完所有城市为止
        while self.move_count < city_num:
            # 移动到下一个城市
            next_city = self.__choice_next_city()
            self.__move(next_city)
        # 计算路径总长度
        self.__cal_total_distance()


# ----------- TSP问题 -----------
class TSP(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.setWindowTitle("TSP蚁群算法")
        self.__r = 5
        self.__lock = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.search_path)
        self.btnInit = QPushButton("初始化")
        self.btnStart = QPushButton("开始搜索")
        self.btnStop = QPushButton("停止搜索")
        self.btnQuit = QPushButton("退出程序")
        self.btnInit.clicked.connect(self.new)
        self.btnStart.clicked.connect(self.start)
        self.btnStop.clicked.connect(self.stop)
        self.btnQuit.clicked.connect(self.quit)
        layout = QVBoxLayout()
        layout.addWidget(self.btnInit)
        layout.addWidget(self.btnStart)
        layout.addWidget(self.btnStop)
        layout.addWidget(self.btnQuit)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.new()

        # 初始化UI

    def initUI(self):
        self.setGeometry(100, 100, 800, 600)
        self.graphicsView = QGraphicsView(self)
        self.setCentralWidget(self.graphicsView)

        # 初始化

    def new(self):
        # 停止搜索
        self.stop()
        # 清除画布
        self.scene.clear()
        self.nodes = []  # 节点坐标
        self.nodes2 = []  # 节点对象
        # 初始化城市节点
        for i in range(len(distance_x)):
            # 在画布上随机初始坐标
            x = distance_x[i]
            y = distance_y[i]
            self.nodes.append((x, y))
            # 生成节点椭圆，半径为self.__r
            node = QGraphicsEllipseItem(x - self.__r, y - self.__r, self.__r * 2, self.__r * 2)
            self.scene.addItem(node)
            self.nodes2.append(node)
            # 显示坐标
            text = QGraphicsTextItem('({},{})'.format(x, y))
            text.setPos(x, y - 10)
            self.scene.addItem(text)
        # 初始城市之间的距离和信息素
        for i in range(city_num):
            for j in range(city_num):
                pheromone_graph[i][j] = 1.0
        self.ants = [Ant(ID) for ID in range(ant_num)]  # 初始蚁群
        self.best_ant = Ant(-1)  # 初始最优解
        self.best_ant.total_distance = 1 << 31  # 初始最大距离
        self.iter = 1  # 初始化迭代次数

        # 开始搜索

    def start(self):
        self.timer.start(100)

        # 停止搜索

    def stop(self):
        self.timer.stop()

        # 退出程序

    def quit(self):
        self.stop()
        self.close()

        # 搜索路径

    def search_path(self):
        # 遍历每一只蚂蚁
        for ant in self.ants:
            # 搜索一条路径
            ant.search_path()
            # 与当前最优蚂蚁比较
            if ant.total_distance < self.best_ant.total_distance:
                # 更新最优解
                self.best_ant = copy.deepcopy(ant)
        # 更新信息素
        self.__update_pheromone_gragh()
        print(u"迭代次数：", self.iter, u"最佳路径总距离：", int(self.best_ant.total_distance))
        # 连线
        self.line(self.best_ant.path)
        # 设置标题
        self.setWindowTitle("TSP蚁群算法 迭代次数: %d" % self.iter)
        # 更新画布
        self.scene.update()
        self.iter += 1

        # 更新信息素

    def __update_pheromone_gragh(self):
        # 获取每只蚂蚁在其路径上留下的信息素
        temp_pheromone = [[0.0 for col in range(city_num)] for raw in range(city_num)]
        for ant in self.ants:
            for i in range(1, city_num):
                start, end = ant.path[i - 1], ant.path[i]
                # 在路径上的每两个相邻城市间留下信息素，与路径总距离反比
                temp_pheromone[start][end] += Q / ant.total_distance
                temp_pheromone[end][start] = temp_pheromone[start][end]
        # 更新所有城市之间的信息素，旧信息素衰减加上新迭代信息素
        for i in range(city_num):
            for j in range(city_num):
                pheromone_graph[i][j] = pheromone_graph[i][j] * RHO + temp_pheromone[i][j]

        # 将节点按order顺序连线

    def line(self, order):
        # 删除原线
        for item in self.scene.items():
            if isinstance(item, QGraphicsLineItem):
                self.scene.removeItem(item)

        def line2(i1, i2):
            p1, p2 = self.nodes[i1], self.nodes[i2]
            line = QGraphicsLineItem(p1[0], p1[1], p2[0], p2[1])
            self.scene.addItem(line)
            return i2

        # order[-1]为初始值
        reduce(line2, order, order[-1])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    tsp = TSP()
    tsp.show()
    sys.exit(app.exec_())

