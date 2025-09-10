import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 你的瞬时系统：dx/dt = f(x, t) = r(t) x (1 - x/K(t)) - c(t) x^2 / (1 + x^2)
def f_of_x_t(x, t, r_t, K_t, c_t):
    return r_t * x * (1.0 - x / K_t) - c_t * x**2 / (1.0 + x**2)

# 替换为你的瞬时参数获取函数（示例为正弦波）
def r_time(t, r0, dr, wr):
    return r0 + dr * np.sin(wr * t)

def K_time(t, K0, dK, wK):
    return K0 + dK * np.sin(wK * t)

def c_time(t, c0, dc, wc):
    return c0 + dc * np.sin(wc * t)

# 计算某个 t 的瞬时平衡点集合（数值解 dx/dt = 0）
# 对一维系统，直接用根查找，或用简单解法近似（此处给出一个简化版本：用网格初值积分收敛到的落点）
def equilibria_at_time(t, r0, K0, c0, dr, wr, dK, wK, dc, wc):
    # 如需严格求解平衡点，可解方程 f(x, t) = 0
    # 这里给出一个简化策略：对若干候选点直接找最近的收敛落点作为代表
    # 定义候选平衡点（仅做示例，实际应通过求解方程得到）
    # 直接用 solve_ivp 以不同初始值收敛到的稳定点作为近似
    rs = r_time(t, r0, dr, wr)
    Ks = K_time(t, K0, dK, wK)
    cs = c_time(t, c0, dc, wc)

    # 网格初值，覆盖合理区间
    x_min, x_max = 0.0, max(1.0, Ks * 1.2)
    grid = np.linspace(x_min, x_max, 100)
    roots = []
    for x0 in grid:
        sol = solve_ivp(lambda x, tt: f_of_x_t(x, tt, rs, Ks, cs),
                        [0, 200], [x0], t_eval=[200], vectorized=True)
        x_end = float(sol.y[0, -1])
        # 简单聚类：将落点按舍入到最近的候选点
        if not np.isnan(x_end):
            roots.append(x_end)
    # 把落点聚成若干簇（这里简单取整数化并去重）
    uniq = sorted(set([round(r, 2) for r in roots]))
    return uniq

def main():
    # 参数设定（示例值）
    r0, K0, c0 = 0.47, 10.0, 1.0
    dr, wr = 0.05, 0.2
    dK, wK = 0.0, 0.0
    dc, wc = 0.0, 0.0

    t_span = (0, 1000)
    t_eval = np.linspace(t_span[0], t_span[1], 1001)

    # 网格初值吸引域随时间的演化（1D）

    x_min, x_max = 0.0, K0 * 1.4
    x_grid = np.linspace(x_min, x_max, 300)

    # 存放每个时间点初值的归属类别（平衡点标签的整数编码）
    domain_over_time = []

    for t in t_eval:
        rs = r_time(t, r0, dr, wr)
        Ks = K_time(t, K0, dK, wK)
        cs = c_time(t, c0, dc, wc)

        # 对网格初值进行前向积分，收敛到的落点（简化聚类）
        # 这里直接用最接近的落点的索引来编码类别
        # 你也可以使用更精确的聚类算法
        categories = []
        # 演示用：我们用一个简单的落点近似聚类
        for x0 in x_grid:
            sol = solve_ivp(lambda x, tt: f_of_x_t(x, tt, rs, Ks, cs),
                            [0, 200], [x0], t_eval=[200])
            x_end = float(sol.y[0, -1])
            # 将 x_end 映射到最近的网格区间索引
            idx = int(np.clip((x_end - x_min) / (x_max - x_min) * (len(x_grid) - 1), 0, len(x_grid)-1))
            categories.append(idx)

        domain_over_time.append(categories)

    # 可视化：BoA 时间演化热力图
    plt.figure(figsize=(8, 4))
    im = plt.imshow(np.array(domain_over_time).T, aspect='auto', origin='lower',
                    extent=[t_span[0], t_span[1], x_min, x_max], cmap='tab20')
    plt.colorbar(im, label='Attractor bin index')
    plt.xlabel('Time t')
    plt.ylabel('Initial x0')
    plt.title('Basin of Attraction evolution under time-periodic parameters')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()