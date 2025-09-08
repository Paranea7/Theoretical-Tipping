import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from numba import njit

# 定义参数
k0 = 10  # k的基础值
dt = 0.01  # 时间步长
time_end = 8000  # 仿真时间
num_steps = int(time_end / dt)

# 定义初值和终值的阈值
initial_threshold = 2.36  # 初值的阈值
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

            # Runge-Kutta 4th order method
            k1 = dt * (r * x * (1 - x / k0) - (x ** 2) / (1 + x ** 2))
            k2 = dt * (r * (x + 0.5 * k1) * (1 - (x + 0.5 * k1) / k0) - ((x + 0.5 * k1) ** 2) / (1 + (x + 0.5 * k1) ** 2))
            k3 = dt * (r * (x + 0.5 * k2) * (1 - (x + 0.5 * k2) / k0) - ((x + 0.5 * k2) ** 2) / (1 + (x + 0.5 * k2) ** 2))
            k4 = dt * (r * (x + k3) * (1 - (x + k3) / k0) - (x + k3) ** 2 / (1 + (x + k3) ** 2))

            x_values[j, i] = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x_values

def calculate_tipping_rate(params):
    w, A = params
    r_func = lambda t: 0.47 + A * np.sin(w * t)
    initial_conditions = np.linspace(0, 5, 2000)
    initial_conditions = initial_conditions[initial_conditions < initial_threshold]

    if len(initial_conditions) == 0:
        return 0  # 返回 0，表示没有初始条件符合要求，避免除以零

    r_values = np.array([r_func(i * dt) for i in range(num_steps)])

    x_values = rk4_optimized(initial_conditions, w, dt, num_steps, r_values)
    final_x = x_values[:, -1]

    n = np.sum(final_x > final_threshold)

    # 进行除法时，确保分母不为零
    tipping_rate = n / len(initial_conditions) if len(initial_conditions) > 0 else 0

    return tipping_rate

# 定义 A 和 w 的范围
A_values = np.linspace(0, 1, 210)  # A的范围
w_values = np.linspace(0.01, 1.10, 210)  # w的范围

# 创建二维 array 用于存储 tipping rate
tipping_rates = np.zeros((len(A_values), len(w_values)))

# 使用 Pool 进行并行计算
if __name__ == "__main__":
    for i, A in enumerate(A_values):
        with Pool() as pool:
            results = pool.map(calculate_tipping_rate, [(w, A) for w in w_values])
            tipping_rates[i, :] = results

    # 绘制热力图
    plt.figure(figsize=(12, 6))
    plt.imshow(tipping_rates, extent=(w_values[0], w_values[-1], A_values[0], A_values[-1]), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Tipping Rate')
    plt.xlabel('$w$')
    plt.ylabel('$A$')
    plt.title('Tipping Rate Heatmap in A-w Space')
    plt.grid()
    plt.tight_layout()
    plt.show()