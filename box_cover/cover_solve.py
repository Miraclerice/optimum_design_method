# -*- coding: utf-8 -*-
# @Author: MiracleRice
# Blog   : miraclerice.com
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sympy import symbols

"""
   变量: [翼板厚度t_f, 高度h]^T = [x1, x2]^T
   单位: mm
   目标函数:
       min[obj_func(x) = 120x1 + x2]
   约束条件:
       cons_func1(x) = -x1 < 0
       cons_func2(x) = -x2 < 0
       cons_func3(x) = 1 - x2 / 40 <= 0
       cons_func4(x) = 1 - (70 * x1 * x2) / 4.5e4 <= 0
       cons_func5(x) = 1 - (7 * x1 ** 3 * x2) / 4.5e5 <= 0
       cons_func6(x) = 1 - (x1 * x2 ** 2) / 3.2e5 <= 0
   注意：opt.minimize的约束条件不等号是反过来的, ineq为不等号>=0
   """


def obj_func(x):
    """目标函数"""
    return 120 * x[0] + x[1]


def cons_func1(x):
    """约束条件，其他同命名函数2-6也为约束条件之一"""
    return x[0]


def cons_func2(x):
    return x[1]


def cons_func3(x):
    return -1 + x[1] / 40


def cons_func4(x):
    return -1 + (70 * x[0] * x[1]) / 4.5e4


def cons_func5(x):
    return -1 + (7 * x[0] ** 3 * x[1]) / 4.5e5


def cons_func6(x):
    return -1 + (x[0] * x[1] ** 2) / 3.2e5


def penalty_func(x, cons_funcs, sign=1):
    """惩罚项"""
    penalty = 0
    for func in cons_funcs:
        # 加入一个偏差，防止除0
        if sign == 1:
            penalty += 1 / (func(x) + 1e-6)
        else:
            penalty -= 1 / (func(x) + 1e-6)
    return penalty


def combined_func(x, r, cons_funcs):
    """总函数值，惩罚项乘以惩罚因子r"""
    return obj_func(x) + r * penalty_func(x, cons_funcs)


def save_log(log):
    """保存迭代日志"""
    with open('log.txt', 'a+', encoding='utf-8') as f:
        f.write(log)


def train(r=3, c=0.7):
    cons_funcs = [cons_func1, cons_func2, cons_func3, cons_func4, cons_func5, cons_func6]
    # 变量初始值
    x = np.array([10., 300.])
    # 惩罚因子
    r = r
    # 降低系数
    c = c

    epsilon1 = 1e-6
    epsilon2 = 1e-6

    iter_k = 0

    constraints = (
        {'type': 'ineq', 'fun': cons_func1},
        {'type': 'ineq', 'fun': cons_func2},
        {'type': 'ineq', 'fun': cons_func3},
        {'type': 'ineq', 'fun': cons_func4},
        {'type': 'ineq', 'fun': cons_func5},
        {'type': 'ineq', 'fun': cons_func6}
    )

    # res = opt.minimize(obj_func, x0=x, method='SLSQP', constraints=constraints)
    # print(res)

    while True:
        iter_k += 1
        # print(f"k: {iter_k:03}, \tx: {x[0]:.4f}, {x[1]:.4f}, \t函数值: {obj_func(x):.4f}, \tr: {r}")
        if iter_k == 1:
            x0, x1 = symbols('x[0], x[1]')
            obj = obj_func([x0, x1])
            log = f"最小化目标函数：{obj}\t降低系数: {c:=.2f} \n" \
                  "迭代次数\t\t变量值\t\t\t\t\t\t目标函数值\t\t\t惩罚因子 \n" \
                  f"k: {iter_k:03}, \tx: [{x[0]:.4f}, {x[1]:.4f}], \t函数值: {obj_func(x):.4f}, \tr: {r}\n"
        else:
            log = f"k: {iter_k:03}, \tx: [{x[0]:.4f}, {x[1]:.4f}], \t\t函数值: {obj_func(x):.4f}, \tr: {r}\n"
        save_log(log)
        x_his = x
        r_his = r
        res = opt.minimize(combined_func, x0=x, args=(r, cons_funcs), method='SLSQP', constraints=constraints)
        x = res.x
        r = r * c

        # 判断终止条件
        f1 = r_his * combined_func(x_his, r_his, cons_funcs)
        f2 = r * combined_func(x, r, cons_funcs)
        if np.linalg.norm(r_his * x_his - r * x) <= epsilon1 and abs(f1 - f2) <= epsilon2:
            # 保存最优解
            save_log(f"k: {iter_k + 1:03}, \tx: [{x[0]:.4f}, {x[1]:.4f}], \t\t函数值: {obj_func(x):.4f}, \tr: {r}\n")
            break
    return x, obj_func(x), r


def experiment_c(r):
    """降低系数c的消融实验"""
    save_log("降低系数c的消融实验\n")
    c = np.arange(0.05, 1, 0.05)
    x_opt = [10., 300.]
    obj_opt = obj_func(x_opt)
    c_opt = 0.7
    r_opt = r
    for i in range(len(c)):
        x, obj, r_res = train(r=r, c=c[i])
        if obj < obj_opt:
            x_opt = x
            obj_opt = obj
            c_opt = c[i]
            r_opt = r_res
    save_log(
        f"降低系数: c={c_opt:.2f}, \t惩罚因子: r={r_opt}, \t最优解: x=[{x_opt[0]:.4f}, {x_opt[1]:.4f}], \t\t函数值: f(x)={obj_opt:.4f}\n\n")
    return x_opt, obj_opt, r_opt, c_opt


def get_r(x, cons_funcs):
    """获取惩罚因子r的初始值, p=2,最后向上取整"""
    r = 2 / 100
    s = abs(obj_func(x) / (penalty_func(x, cons_funcs, sign=-1) + 1e-6))
    return round(r * s)


def main():
    r = get_r([10., 30.], cons_funcs=[cons_func1, cons_func2, cons_func3, cons_func4, cons_func5, cons_func6])
    # experiment_c(r)
    train()


if __name__ == '__main__':
    main()
