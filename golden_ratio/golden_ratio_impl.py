# -*- coding: utf-8 -*-
# @Author: MiracleRice
# Blog   : miraclerice.com

import time
import numpy as np
from matplotlib import pyplot as plt
from sympy import symbols

# 用来正常显示中文标签
# plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号 # python2在中文字符前加入u表示unicode编码,python3默认unicode编码
plt.rcParams['axes.unicode_minus'] = False
# 设置全局字体大小
plt.rcParams['font.size'] = 14


def save_log(log):
    """保存迭代日志"""
    with open('log.txt', 'a+', encoding='utf-8') as f:
        f.write(log)


class Goldenratio(object):
    """
    用黄金分割法求解最优点
    :parameter
        lvalue: 区间左端点
        rvalue: 区间右端点
        func: 需要求解最优点的函数
        ratio: 黄金分割比
        epsilon: 收敛精度
    """

    def __init__(self,
                 lvalue=0.,
                 rvalue=1.,
                 func=lambda x: x ** 2 - 7 * x + 10,
                 ratio=0.618,
                 epsilon=1e-6):
        self.lvalue = lvalue
        self.rvalue = rvalue
        self.func = func
        self.ratio = ratio
        self.epsilon = epsilon

    def __call__(self, func_str, plot=False, *args, **kwargs):
        opt = self.res_show(func_str)
        # 根据plot判断是否描绘
        if plot:
            self.plot_show(opt)

    def extremum(self, func, epsilon=None):
        """
        求解极值点
        :param
            func: 需要求解最优点的函数
        """
        # 根据黄金分割比求得新区间
        lvalue, rvalue = self.lvalue, self.rvalue
        lvalue_tmp = rvalue - (rvalue - lvalue) * self.ratio
        rvalue_tmp = lvalue + (rvalue - lvalue) * self.ratio
        f1, f2 = func(lvalue_tmp), func(rvalue_tmp)
        # 初始化搜索迭代次数和记录搜索日志
        iter_epoch, iter_log = 0, []

        # 判断是否终止迭代
        while abs(rvalue - lvalue) > epsilon:
            # 根据区间值求的函数值大小比较，确定新区间
            if f1 < f2:
                iter_epoch += 1
                rvalue = rvalue_tmp
                rvalue_tmp = lvalue_tmp
                f2 = f1
                lvalue_tmp = rvalue - (rvalue - lvalue) * self.ratio
                f1 = func(lvalue_tmp)
                iter_log.append(f"第{iter_epoch:03}次迭代后新区间为: [{lvalue:.3f}, {rvalue:.3f}]")

            else:
                iter_epoch += 1
                lvalue = lvalue_tmp
                lvalue_tmp = rvalue_tmp
                f1 = f2
                rvalue_tmp = lvalue + (rvalue - lvalue) * self.ratio
                f2 = func(rvalue_tmp)
                iter_log.append(f"第{iter_epoch:03}次迭代后新区间为: [{lvalue:.3f}, {rvalue:.3f}]")

        return (lvalue + rvalue) / 2, iter_log

    def res_show(self, func_str, epsilon=None):
        func = self.func
        # 判断是否修改了epsilon
        epsilon = self.epsilon if epsilon is None else epsilon
        extremum, iter_log = self.extremum(func, epsilon)
        func_info = ("已知条件".center(70, "*")
                     + f"\n函数表达式:{func_str}\t区间: [{self.lvalue:.3f}, {self.rvalue:.3f}]\t收敛精度: {epsilon}")
        res = f"黄金分割法下的最优点为x* = {extremum:.3f} \t f(x*) = {func(extremum):.3f}"
        local_t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        save_log(f"{local_t}".center(72, "*"))
        save_log(f"\n{func_info}\n")
        save_log("迭代日志".center(70, "*") + "\n")
        for i in iter_log:
            save_log(f"{i}\n")
        save_log("最优结果".center(70, "*"))
        save_log(f"\n{res}\n\n\n")
        return extremum

    def plot_show(self, optimal):
        x = np.linspace(self.lvalue, self.rvalue, 100)
        y = self.func(x)
        plt.plot(x, y, c="b")
        plt.scatter(optimal, self.func(optimal), c="r")
        plt.annotate(
            text=f'x*={optimal:.3f}',  # 注释内容
            xy=(optimal, self.func(optimal)),  # 注释的坐标点，也就是箭头指向的位置
            xytext=(2.5, 1.5),  # 标注内容的位置
            # 箭头样式
            arrowprops={
                'width': 2,  # 箭头线的宽度l
                'headwidth': 5,  # 箭头头部的宽度
                'facecolor': 'yellow'  # 箭头的背景颜色
            }
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Golden Ratio Method")
        plt.savefig("golden_ratio.png", dpi=300)


def experiment_epsilon():
    """修改收敛精度，获取极值"""
    save_log("消融实验".center(70, "*") + "\n")
    ratio, lvalue, rvalue, epsilon, func, func_str = init_data()
    # 收敛精度降低系数
    # epsilon = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035]
    epsilon = np.arange(0.005, 0.040, 0.005)
    for i in range(len(epsilon)):
        golden_ratio = Goldenratio(lvalue, rvalue, func, ratio, epsilon[i])
        # 只描绘第一个
        plot = True if i == 0 else False
        golden_ratio(func_str, plot)


def init_data():
    """初始化已知条件"""
    # 黄金分割比例
    ratio = (-1 + 5 ** 0.5) / 2
    lvalue = float(input("请输入区间左值:"))
    rvalue = float(input("请输入区间右值:"))
    epsilon = float(input("请输入收敛精度:"))
    func = lambda x: x ** 2 - 7 * x + 10
    x = symbols("x")
    # func_str = "f(x) = x ** 2 - 7 * x + 10"
    func_str = f"f(x) = {func(x)}"
    return ratio, lvalue, rvalue, epsilon, func, func_str


def main():
    # ratio, lvalue, rvalue, epsilon, func, func_str = init_data()
    # golden_ratio = Goldenratio(lvalue, rvalue, func, ratio, epsilon)
    # golden_ratio(func_str)
    # 实验
    experiment_epsilon()


if __name__ == '__main__':
    main()
