import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint



def k_strategy(y, t, s, K, alpha, beta):
    N, P = y
    dNdt = s * N * (1 - (N + alpha * P)/K)
    dPdt = s * P * (1 - (P + beta * N)/K)
    return [dNdt, dPdt]

# 参数
s, K, alpha, beta = 0.1, 5000, 0.8, 1.2
t = np.linspace(0, 100, 1000)
y0 = [100, 100]  # 初始种群

# 求解微分方程
solution = odeint(k_strategy, y0, t, args=(s, K, alpha, beta))
N, P = solution.T
plt.plot(t, N, label="Species N")
plt.plot(t, P, label="Species P")
plt.xlabel("Time"); plt.ylabel("Population"); plt.title("K");
plt.legend(); plt.show()