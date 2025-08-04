import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from numba import njit

# 定义参数
r = 0.47
dt = 0.01  # 时间步长
time_end = 100  # 仿真时间
num_steps = int(time_end / dt)

# 模型参数
alpha = 1.0  # 参数 alpha
beta = 1.0  # 参数 beta

# 定义初值和终值的阈值
initial_threshold = 2.36  # 初值的阈值
final_threshold = 3.5  # 终值的阈值

@njit
def rk4_optimized(x0s, w, dt, num_steps, K_values):
    num_initial_conditions = len(x0s)
    x_values = np.zeros((num_initial_conditions, num_steps))  # 存储每个初始条件的 x 值
    x_values[:, 0] = x0s  # 设置初始值

    for i in range(1, num_steps):
        k0 = K_values[i]  # 从预计算的 r 值数组中获取 r

        for j in range(num_initial_conditions):
            x = x_values[j, i - 1]  # 获取前一步的 x 值

            k1 = dt * (r * x * (1 - x / k0) - (beta * x ** 2) / (alpha ** 2 + x ** 2))
            k2 = dt * (r * (x + 0.5 * k1) * (1 - (x + 0.5 * k1) / k0) -
                       (beta * (x + 0.5 * k1) ** 2) / (alpha ** 2 + (x + 0.5 * k1) ** 2))
            k3 = dt * (r * (x + 0.5 * k2) * (1 - (x + 0.5 * k2) / k0) -
                       (beta * (x + 0.5 * k2) ** 2) / (alpha ** 2 + (x + 0.5 * k2) ** 2))
            k4 = dt * (r * (x + k3) * (1 - (x + k3) / k0) -
                       (beta * (x + k3) ** 2) / (alpha ** 2 + (x + k3) ** 2))

            x_values[j, i] = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x_values

# 计算 tipping rate 的函数
def calculate_tipping_rate(params):
    w, r_func = params
    initial_conditions = np.linspace(0, 4.73, 2000)  # 初始条件范围
    initial_conditions = initial_conditions[initial_conditions < initial_threshold]  # 过滤初值符合条件的初始条件

    if len(initial_conditions) == 0:  # 如果没有符合初值条件的初始条件
        return 0

    # 计算预计算的 r 值
    K_values = np.array([r_func(w, i * dt) for i in range(num_steps)])

    x_values = rk4_optimized(initial_conditions, w, dt, num_steps, K_values)  # 使用加速的RK4求解 x 的值
    final_x = x_values[:, -1]  # 取每个初始条件最终的x值

    n = np.sum(final_x > final_threshold)  # 计算满足终值条件的计数
    tipping_rate = n / len(initial_conditions)  # 计算 tipping rate

    return tipping_rate

# 定义振幅函数
def r_func1(w, t):
    return 10 + np.exp(-t) * np.sin(w * t)

def r_func2(w, t):
    return 10 + 1 / (t + 1e-18) * np.sin(w * t)  # 选择一个非常小的epsilon来避免除以零

def r_func3(w, t):
    return 10 + 0.5 * np.sin(w * t)

def r_func4(w, t):
    return 10 + 1 * np.sin(w * t)  # 新加的振幅函数，振幅为0.1

def r_func5(w, t):
    return 10 + 2 * np.sin(w * t)  # 新加的振幅函数，振幅为0.2

# w 的范围
w_values = np.linspace(0.01, 1.10, 1110)  # w的范围

# 使用 Pool 进行并行计算并绘制每个振幅对应的 tipping rate
if __name__ == "__main__":
    plt.figure(figsize=(12, 6))

    r_funcs = [r_func1, r_func2, r_func3, r_func4, r_func5]  # 更新振幅函数的列表，包括新添加的函数
    A_labels = ['A(t) = e^{-t}sin(wt)', 'A(t) = 1/(t+ε)', 'A(t) = 0.05sin(wt)', 'A(t) = 0.1sin(wt)', 'A(t) = 0.2sin(wt)']

    for r_func, label in zip(r_funcs, A_labels):
        with Pool() as pool:
            results = pool.map(calculate_tipping_rate, [(w, r_func) for w in w_values])

        plt.plot(w_values, results, marker='o', linestyle='-', label=label)

    plt.xlabel('$w$')
    plt.ylabel('Tipping Rate')
    plt.title('Tipping Rate vs. $w$ for Different Amplitudes')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()