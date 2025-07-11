import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 设置 Matplotlib 支持中文
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 1. 定义系统方程
def system(x, r_x, K):
    dxdt = r_x * x * (1 - x / K) - (x ** 2) / (1 + x ** 2)
    return dxdt


# 2. 寻找平衡点
def find_equilibrium(r_x, K, initial_guess=1.0):
    def equilibrium_eq(x):
        return system(x, r_x, K)

    equilibria = [0.0]  # 总包含零解
    try:
        x_nonzero = fsolve(equilibrium_eq, initial_guess)[0]
        if abs(x_nonzero) > 1e-6:  # 排除数值误差
            equilibria.append(x_nonzero)
    except Exception as e:
        print(f"警告: 寻找非零平衡点时出错 - {e}")
    return np.array(equilibria)


# 3. 计算雅可比矩阵（导数）
def jacobian(x, r_x, K):
    dJ = r_x * (1 - 2 * x / K) - (2 * x * (1 + x ** 2) - 2 * x ** 3) / (1 + x ** 2) ** 2
    return dJ


# 4. 数值模拟函数
def simulate_system(r_x, K, t_span=(0, 100), x0=0.1):
    def ode_func(t, x):
        return system(x, r_x, K)

    sol = solve_ivp(ode_func, t_span, [x0], method='LSODA', rtol=1e-6)
    return sol.t, sol.y[0]


# 5. 主程序
if __name__ == "__main__":
    # 参数设置
    K = 10.0  # 固定承载能力 K 值
    r_x = 0.47  # 初始生长率 r

    # 步骤1: 计算平衡点
    equilibria = find_equilibrium(r_x, K)
    print("平衡点:", equilibria)

    # 步骤2: 分析稳定性
    for x_eq in equilibria:
        J = jacobian(x_eq, r_x, K)
        stability = "稳定" if J < 0 else "不稳定"
        print(f"平衡点 {x_eq:.3f}: J = {J:.3f} ({stability})")

    # 步骤3: 数值模拟
    t, x = simulate_system(r_x=r_x, K=K, x0=0.5)
    plt.figure()
    plt.plot(t, x)
    plt.xlabel('时间')
    plt.ylabel('x')
    plt.title('动态模拟')
    plt.grid()
    plt.show()

    # 步骤4: 分岔分析（参数 K 变化）
    K_values = np.linspace(1, 15, 500)
    final_states_k = []
    for K_param in K_values:
        t, x = simulate_system(r_x=r_x, K=K_param, t_span=(0, 100), x0=0.1)
        final_states_k.append(x[-1])

    plt.figure()
    plt.plot(K_values, final_states_k, '.', markersize=2)
    plt.xlabel('K (承载能力)')
    plt.ylabel('最终状态 x')
    plt.title('分岔图（K变化）')
    plt.grid()
    plt.show()

    # 步骤5: 分岔分析（参数 r_x 变化）
    r_values = np.linspace(0.1, 1.0, 500)
    final_states_r = []
    for r_param in r_values:
        t, x = simulate_system(r_x=r_param, K=K, t_span=(0, 100), x0=0.1)
        final_states_r.append(x[-1])

    plt.figure()
    plt.plot(r_values, final_states_r, '.', markersize=2)
    plt.xlabel('r_x (生长率)')
    plt.ylabel('最终状态 x')
    plt.title('分岔图（r变化）')
    plt.grid()
    plt.show()

    # 步骤6: 绘制 r_x 和 K 的相平面
    r_x_values = np.linspace(0.0, 1.0, 500)
    K_values = np.linspace(0, 20, 500)
    stability_map = []

    for r_param in r_x_values:
        row = []
        for K_param in K_values:
            equilibria = find_equilibrium(r_param, K_param)
            stability = '稳定' if any(jacobian(x_eq, r_param, K_param) < 0 for x_eq in equilibria) else '不稳定'
            row.append(stability)
        stability_map.append(row)

    stability_map = np.array(stability_map)

    # 绘制相平面
    plt.figure(figsize=(8, 6))
    plt.imshow(stability_map == '稳定', extent=(K_values.min(), K_values.max(), r_x_values.min(), r_x_values.max()),
               origin='lower', aspect='auto', cmap='Greys', alpha=0.3)
    plt.colorbar(label='稳定性（黑色为稳定）')
    plt.scatter([], [], color='black', label='稳定')  # 为图例添加一个标签
    plt.scatter([], [], color='white', label='不稳定')  # 为图例添加一个标签

    plt.xlabel('K (承载能力)')
    plt.ylabel('r_x (生长率)')
    plt.title('r_x 和 K 的相平面')
    plt.legend()
    plt.grid()
    plt.show()