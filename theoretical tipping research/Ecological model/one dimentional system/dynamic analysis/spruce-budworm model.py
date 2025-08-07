import numpy as np
import matplotlib.pyplot as plt

# 定义参数
k = 10
alpha = 1
beta = 1
dt = 0.01
time_end = 100
num_steps = int(time_end / dt)

# RK4 方法
def rk4(x0, r_value, dt, num_steps):
    x = x0
    for _ in range(num_steps):
        k1 = dt * (r_value * x * (1 - x / k) - (beta * x ** 2) / (alpha ** 2 + x ** 2))
        k2 = dt * (r_value * (x + 0.5 * k1) * (1 - (x + 0.5 * k1) / k) -
                   (beta * (x + 0.5 * k1) ** 2) / (alpha ** 2 + (x + 0.5 * k1) ** 2))
        k3 = dt * (r_value * (x + 0.5 * k2) * (1 - (x + 0.5 * k2) / k) -
                   (beta * (x + 0.5 * k2) ** 2) / (alpha ** 2 + (x + 0.5 * k2) ** 2))
        k4 = dt * (r_value * (x + k3) * (1 - (x + k3) / k) -
                   (beta * (x + k3) ** 2) / (alpha ** 2 + (x + k3) ** 2))
        x += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x

# 生成 r 值，范围从 0 到 1，步长为 0.02
r_values = np.linspace(0, 1, 50)
X_initial_conditions = [0.1, 5]  # 假设我们从这两个初始条件出发

x_stable = []
roots_np = []

# 对每个 r 值分别从不同初始条件计算稳态 x
for r_value in r_values:
    for x0 in X_initial_conditions:
        steady_x = rk4(x0=x0, r_value=r_value, dt=dt, num_steps=num_steps)
        x_stable.append((r_value, steady_x, x0))

    roots = np.roots([-r_value, k*r_value, -(k+r_value), k*r_value])
    real_roots = roots[np.isreal(roots)].real
    for root in real_roots:
        roots_np.append((r_value, root))

# 将结果分离为 r 和 x 的列表
r_plot = []
x_plot = []

# 根据不同的初始条件合并 x 值
for r_value, steady_x, x0 in x_stable:
    r_plot.append(r_value)
    x_plot.append(steady_x)

# 将相同的 r 值分为不同的 x 值
r_unique = sorted(set(r_plot))
x_bistable_1 = []
x_bistable_2 = []

for r in r_unique:
    x_vals = [x_plot[i] for i in range(len(x_plot)) if r_plot[i] == r]
    if len(x_vals) > 0:
        # 选择最大和最小的 x 值作为双稳态输出
        x_bistable_1.append(max(x_vals))
        x_bistable_2.append(min(x_vals))

# 将找到的根分离为 r 和 x 的数组
r_points_np, x_points_np = zip(*roots_np) if roots_np else ([], [])

# 绘制双稳态曲线
plt.figure(figsize=(12, 6))
# 使用 np.roots 方法得到的不动点
plt.scatter(r_points_np, x_points_np, s=10, c='y', zorder=2)
plt.plot(r_unique, x_bistable_1, marker='o', color='blue', zorder=1)
plt.plot(r_unique, x_bistable_2, marker='x', color='red', zorder=1)
plt.xlabel('r')
plt.ylabel('Steady state x')
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 11, 1))

plt.grid()
plt.show()