import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def r_strategy(N, t, r, K, epsilon, L):
    dNdt = r * N * (1 - N/K)
    if N >= L:
        dNdt -= epsilon * N**2
    return dNdt

# 参数
r, K, epsilon, L = 2.0, 1000, 0.01, 800
t = np.linspace(0, 10, 1000)
N0 = 10  # 初始种群

# 求解微分方程
solution = odeint(r_strategy, N0, t, args=(r, K, epsilon, L))
plt.plot(t, solution, label="Population")
plt.xlabel("Time"); plt.ylabel("N"); plt.title("r");
plt.show()