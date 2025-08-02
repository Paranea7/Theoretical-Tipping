import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from numba import njit

# 定义参数
k0 = 10  # k的基础值
dt = 0.01  # 时间步长
time_end = 100  # 仿真时间
num_steps = int(time_end / dt)

# 模型参数
alpha = 1.0  # 参数 alpha
beta = 1.0  # 参数 beta

# 定义初值和终值的阈值
initial_threshold = 2.32  # 初值的阈值
final_threshold = 3  # 终值的阈值

@njit
def rk4_optimized(x0s, w, dt, num_steps, r_values):
    num_initial_conditions = len(x0s)
    x_values = np.zeros((num_initial_conditions, num_steps))  # 存储每个初始条件的 x 值
    x_values[:, 0] = x0s  # 设置初始值

    for i in range(1, num_steps):
        r = r_values[i]  # 从预计算的 r 值数组中获取 r

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
    r_values = np.array([r_func(w, i * dt) for i in range(num_steps)])

    x_values = rk4_optimized(initial_conditions, w, dt, num_steps, r_values)  # 使用加速的RK4求解 x 的值
    final_x = x_values[:, -1]  # 取每个初始条件最终的x值

    n = np.sum(final_x > final_threshold)  # 计算满足终值条件的计数
    tipping_rate = n / len(initial_conditions)  # 计算 tipping rate

    return tipping_rate

# 定义振幅函数
def r_func1(w, t):
    return 0.47 + np.exp(-t) * np.sin(w * t)

def r_func2(w, t):
    return 0.47 + 1 / (t + 1e-18) * np.sin(w * t)  # 选择一个非常小的epsilon来避免除以零

def r_func3(w, t):
    return 0.47 + 0.05 * np.sin(w * t)

# w 和振幅的范围
w_values = np.linspace(0.01, 1.10, 1110)  # w的范围
A_values = np.array([0.47 + np.exp(-t) for t in np.linspace(0, 20, 100)])  # 振幅范围，简单选择

# 创建一个矩阵保存 tipping rate
tipping_rate_matrix = np.zeros((len(A_values), len(w_values)))

# 使用 Pool 进行并行计算填充 tipping_rate_matrix
if __name__ == "__main__":
    for i, r_func in enumerate([r_func1, r_func2, r_func3]):
        with Pool() as pool:
            results = pool.map(calculate_tipping_rate, [(w, r_func) for w in w_values])

        # 填充矩阵
        tipping_rate_matrix[:, :] = np.array(results).reshape(len(A_values), -1)

    # 绘制热力图
    plt.figure(figsize=(10, 6))
    plt.imshow(tipping_rate_matrix, aspect='auto', origin='lower',
               extent=[w_values[0], w_values[-1], A_values[0], A_values[-1]],
               cmap='viridis', vmin=0, vmax=1)  # vmin和vmax分别是tipping rate的范围

    plt.colorbar(label='Tipping Rate')
    plt.xlabel('$w$')
    plt.ylabel('Amplitude')
    plt.title('Tipping Rate Heatmap as a function of Amplitude and $w$')
    plt.tight_layout()
    plt.show()