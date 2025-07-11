import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
from matplotlib.colors import ListedColormap

# 定义模型
def model(r, t, arg):
    B, S = r
    dBdt = arg[0] * B * (1 - B / (4. * S)) - (arg[1] * (B ** 2) / (S ** 2 + B ** 2))
    dSdt = arg[2] * S * (1 - S / arg[3]) - arg[4] * B
    return [dBdt, dSdt]

# 定义平衡方程
def equilibrium(r, arg):
    B, S = r
    eq1 = arg[0] * B * (1 - B / (4. * S)) - (arg[1] * (B ** 2) / (S ** 2 + B ** 2))
    eq2 = arg[2] * S * (1 - S / arg[3]) - arg[4] * B
    return [eq1, eq2]

# 参数设置
arg = [0.1, 0.02, 0.1, 100, 0.01]
t = np.linspace(0, 100, 1000)

# 计算吸引点
initial_guesses = [[0.1, 1], [10, 50], [20, 25], [30, 15]]  # 预先定义的初始猜测
fixed_points = []

for guess in initial_guesses:
    point = fsolve(equilibrium, guess, args=(arg,))
    if point[0] >= 0 and point[1] >= 0:
        fixed_points.append(point)

fixed_points = np.array(fixed_points)

# 创建网格点
B_values = np.linspace(0.1, 300, 1000)  # B的范围
S_values = np.linspace(0.1, 1000, 1000)  # S的范围
B_grid, S_grid = np.meshgrid(B_values, S_values)  # 网格化

# 存储吸引点结果
attraction = np.zeros(B_grid.shape)

# 评估每个网格点的吸引情况
for i in range(B_grid.shape[0]):
    for j in range(B_grid.shape[1]):
        r0 = [B_grid[i, j], S_grid[i, j]]  # 初始条件
        solution = odeint(model, r0, t, args=(arg,))
        final_state = solution[-1]  # 最后一个状态
        # 找到最近的吸引点
        distances = np.linalg.norm(fixed_points - final_state, axis=1)
        closest_fp_index = np.argmin(distances)  # 找到离最终状态最近的吸引点
        attraction[i, j] = closest_fp_index  # 记录吸引点索引

# 绘制吸引域
plt.figure(figsize=(10, 8))
cmap = ListedColormap(['lightblue', 'lightgreen', 'salmon'])  # 自定义颜色图
plt.contourf(B_grid, S_grid, attraction, levels=np.arange(len(fixed_points) + 1) - 0.5, cmap=cmap)
plt.colorbar(ticks=range(len(fixed_points)), label='Attractor Index')
plt.scatter(fixed_points[:, 0], fixed_points[:, 1], color='black', marker='x', s=100, label='Fixed Points')
plt.title('Basin of Attraction')
plt.xlabel('Population B')
plt.ylabel('Population S')
plt.legend()
plt.grid()
plt.show()