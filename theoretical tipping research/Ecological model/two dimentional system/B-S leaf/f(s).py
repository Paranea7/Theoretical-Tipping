import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 参数设置
r_B = 0.2
r_S = r_B/100.
K_S = 4000
beta = 200
e_B = 0.002

# 函数定义
def f(S):
    return 4. * S # 示例函数

def g(S):
    return S/2.  # 示例函数

# 模型方程
def model(y, t):
    B, S = y
    dBdt = r_B * B * (1 - B / f(S)) - (beta * B**2 / (g(S)**2 + B**2))
    dSdt = r_S * S * (1 - S / K_S) - e_B * B
    return [dBdt, dSdt]

# 初始条件
B0 = 5000
S0 = 1000
y0 = [B0, S0]

# 时间点
t = np.linspace(0, 1000, 10000)

# 数值解
solution = odeint(model, y0, t)
B, S = solution.T

# 绘制演化图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t, B, label='B (Species)', color='blue')
plt.plot(t, S, label='S (Resource)', color='green')
plt.xlabel('Time')
plt.ylabel('Population / Resource')
plt.legend()
plt.title('Evolution Plot')

# 绘制相图
plt.subplot(1, 2, 2)
plt.plot(B, S, label='Phase Space', color='purple')
plt.xlabel('B (Species)')
plt.ylabel('S (Resource)')
plt.title('Phase Diagram')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()