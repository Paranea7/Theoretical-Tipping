import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation

# 参数设置
V_c = 10.0
D = 0.001
dt = 0.01
dx = dy = 0.1
nx = ny = 50  # 网格维度
c_values = [1.14]  # 选择一个 c 值
num_iterations = 6000  # 迭代次数
last_steps = 1000  # 最后绘制的步数

# 随时间变化的 r
def r(t):
    return 0.47 + 0.05 * np.sin(0.01 * t)  # 随时间 t 线性变化

# 不随时间变化的 r
def r_static():
    return 0.47  # 静态 r 值

# 定义拉普拉斯算子
def laplacian(V):
    return (np.roll(V, 1, axis=0) + np.roll(V, -1, axis=0) +
            np.roll(V, 1, axis=1) + np.roll(V, -1, axis=1) - 4 * V) / (dx * dy)

# RK4迭代算法的向量化实现
def rk4_step(V, r_value, V_c, c):
    k1 = (r_value * V * (1 - V / V_c) - c * V ** 2 / (V ** 2 + 1) + D * laplacian(V))
    k2 = (r_value * (V + 0.5 * dt * k1) * (1 - (V + 0.5 * dt * k1) / V_c) -
          c * (V + 0.5 * dt * k1) ** 2 / ((V + 0.5 * dt * k1) ** 2 + 1) +
          D * laplacian(V + 0.5 * dt * k1))
    k3 = (r_value * (V + 0.5 * dt * k2) * (1 - (V + 0.5 * dt * k2) / V_c) -
          c * (V + 0.5 * dt * k2) ** 2 / ((V + 0.5 * dt * k2) ** 2 + 1) +
          D * laplacian(V + 0.5 * dt * k2))
    k4 = (r_value * (V + dt * k3) * (1 - (V + dt * k3) / V_c) -
          c * (V + dt * k3) ** 2 / ((V + dt * k3) ** 2 + 1) +
          D * laplacian(V + dt * k3))

    V_next = V + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return V_next

# 从 CSV 文件读取 V
V_initial = np.loadtxt('1.5-5.5.csv', delimiter=',')
frames_static = []
frames_dynamic = []
frames_diff = []

# 在最后1000步中进行迭代并更新索引
for t in range(num_iterations):
    # 只有前num_iterations - last_steps步计算静态frame
    if t < num_iterations - last_steps:
        V_initial = rk4_step(V_initial, r_static(), V_c, c_values[0])
        if t >= 0:  # 记录静态结果
            frames_static.append(V_initial.copy())
    # 动态r
    V_dynamic = rk4_step(V_initial.copy(), r(t), V_c, c_values[0])
    frames_dynamic.append(V_dynamic.copy())

    # 计算差分图（差值只记录最后1000步）
    if t >= num_iterations - last_steps:
        V_diff = V_dynamic - frames_static[-1]  # 取静态最后一帧进行差分
        frames_diff.append(V_diff.copy())

# 只保留最后1000步的动态和差分图
frames_dynamic = frames_dynamic[-last_steps:]
frames_diff = frames_diff[-last_steps:]

# 创建图像以呈现动画
fig, axs = plt.subplots(1, 3, figsize=(24, 8))
(im_static, im_dynamic, im_diff) = (
    axs[0].imshow(frames_static[-1], extent=[0, nx * dx, 0, ny * dy], origin='lower',
                  aspect='auto', vmin=0, vmax=6),
    axs[1].imshow(frames_dynamic[0], extent=[0, nx * dx, 0, ny * dy], origin='lower',
                  aspect='auto', vmin=0, vmax=6),
    axs[2].imshow(frames_diff[0], extent=[0, nx * dx, 0, ny * dy], origin='lower',
                  aspect='auto', vmin=-2, vmax=2)
)

# 设置标题和标签
axs[0].set_title('Static r: c = {}'.format(c_values[0]))
axs[1].set_title('Dynamic r: c = {}'.format(c_values[0]))
axs[2].set_title('Difference: c = {}'.format(c_values[0]))

for ax in axs:
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im_static, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)

# 更新函数
def update(frame):
    im_dynamic.set_array(frames_dynamic[frame])
    im_diff.set_array(frames_diff[frame])
    return [im_dynamic, im_diff]

# 创建动画，减慢播放速度
ani = FuncAnimation(fig, update, frames=len(frames_dynamic), blit=True, repeat=False, interval=10)  # 设置interval为100ms

# 保存为GIF文件
ani.save('simulation_animation.gif', writer='imagemagick', fps=10)  # fps设置为10

plt.show()