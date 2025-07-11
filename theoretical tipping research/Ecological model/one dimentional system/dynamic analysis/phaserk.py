import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.integrate import solve_ivp

# 参数设置
K_min, K_max = 0.1, 20.0  # x轴
r_min, r_max = 0.0, 1.0  # y轴
resolution = 250
threshold = 1.5  # 区分Refuge和Outbreak的阈值

# 生成参数网格
K = np.linspace(K_min, K_max, resolution)
r = np.linspace(r_min, r_max, resolution)
K_grid, R_grid = np.meshgrid(K, r)

# 初始化相图矩阵
phase_matrix = np.zeros_like(K_grid, dtype=int)


def analyze_system(current_r, current_K):
    """综合稳定性分析与稳态值检测"""
    # 解三次方程求所有稳态
    coeff = [current_r, -current_r * current_K, current_r + current_K, -current_r * current_K]
    roots = np.roots(coeff)

    # 收集有效正实根
    valid_roots = []
    for root in roots:
        if abs(root.imag) < 1e-6 and root.real >= 0:
            valid_roots.append(root.real)
    valid_roots = sorted(list(set(valid_roots)))

    # 稳定性分析
    stable_states = []
    for x in [0.0] + valid_roots:
        if x < 1e-6:
            df = current_r
        else:
            term1 = current_r * (1 - 2 * x / current_K)
            term2 = (2 * x) / (1 + x ** 2) ** 2
            df = term1 - term2
        if df < -1e-6:
            stable_states.append(x)

    # 分类系统状态
    if not stable_states:
        return 0  # 灭绝
    elif all(x < 1e-3 for x in stable_states):
        return 0  # 灭绝

    if len(stable_states) > 1:
        return 3  # 双稳态

    # 单稳态时判断具体类型
    x_star = stable_states[0]

    # 附加验证：使用RK4积分确认最终状态
    def ode(t, x):
        return current_r * x * (1 - x / current_K) - x ** 2 / (1 + x ** 2)

    sol = solve_ivp(ode, [0, 100], [0.5], method='RK45', rtol=1e-6)
    final_state = sol.y[0, -1]

    # 结合解析解和数值解判断
    if (x_star > threshold) or (final_state > threshold):
        return 2  # Outbreak
    else:
        return 1  # Refuge


# 向量化计算（注意会大幅增加计算时间）
vector_analyzer = np.vectorize(analyze_system)
phase_matrix = vector_analyzer(R_grid, K_grid)

# 创建增强颜色映射
cmap = ListedColormap([
    '#2C3E50',  # 0: 灭绝
    '#27AE60',  # 1: Refuge
    '#2980B9',  # 2: Outbreak
    '#E74C3C'  # 3: 双稳态
])

# 绘制相图
plt.figure(figsize=(12, 8))
plt.pcolormesh(K_grid, R_grid, phase_matrix,
               cmap=cmap, shading='auto', edgecolors='face', linewidth=0.1)

# 添加分岔曲线和标注
x_bif = np.linspace(1.01, 4, 1000)
r_bif = 2 * x_bif ** 3 / (1 + x_bif ** 2) ** 2
K_bif = 2 * x_bif ** 3 / (x_bif ** 2 - 1)
valid = (r_bif >= r_min) & (r_bif <= r_max) & (K_bif >= K_min) & (K_bif <= K_max)
plt.plot(K_bif[valid], r_bif[valid], 'w--', lw=1.5, alpha=0.8)

# 图形标注
plt.text(15, 0.8, 'Outbreak Zone', color='white', ha='center',
         fontsize=12, bbox=dict(facecolor='#2980B980', edgecolor='none'))
plt.text(3, 0.2, 'Refuge Zone', color='white', ha='center',
         fontsize=12, bbox=dict(facecolor='#27AE6080', edgecolor='none'))
plt.text(8, 0.6, 'Bistable', color='white', ha='center', rotation=45)

# 坐标轴设置
plt.xlabel('Carrying Capacity (K)', fontsize=12)
plt.ylabel('Growth Rate (r)', fontsize=12)
plt.title('Enhanced Phase Diagram with Dynamic State Verification', pad=20)
cbar = plt.colorbar(ticks=[0, 1, 2, 3])
cbar.ax.set_yticklabels(['Extinction', 'Refuge', 'Outbreak', 'Bistable'])
plt.grid(alpha=0.2, linestyle=':')
plt.xlim(K_min, K_max)
plt.ylim(r_min, r_max)
plt.show()