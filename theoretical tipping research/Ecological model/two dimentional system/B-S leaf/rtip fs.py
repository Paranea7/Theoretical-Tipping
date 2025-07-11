import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 参数设置
r_B_base = 0.2  # 基础生长率
K_S = 4000  # 资源的环境承载力
beta = 200  # 竞争系数
e_B = 0.002  # 消耗资源的比率
omega = 0.05  # 频率
A_B = 0.02  # r_B 的幅度

# 函数定义
def f(S):
    return 4. * S  # 示例函数，定义物种的资源获取上限

def g(S):
    return S / 2.  # 示例函数，与物种竞争相关

# 模型方程，包括时间依赖的 r_B 和 r_S
def model(y, t):
    B, S = y
    r_B = r_B_base + A_B * np.sin(omega * t)  # 时间相关的 r_B
    r_S = r_B / 100.  # 资源增长率为 r_B 的百分之一
    dBdt = r_B * B * (1 - B / f(S)) - (beta * B**2 / (g(S)**2 + B**2))  # 物种增长
    dSdt = r_S * S * (1 - S / K_S) - e_B * B  # 资源变化
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

# 绘制物种与资源的演变曲线
plt.subplot(1, 2, 1)
plt.plot(t, B, label='B (Species)', color='blue')
plt.plot(t, S, label='S (Resource)', color='green')
plt.xlabel('Time')
plt.ylabel('Population / Resource')
plt.legend()
plt.title('Evolution Plot with Time-Dependent r_B and r_S')

# 绘制相图
plt.subplot(1, 2, 2)
plt.plot(B, S, label='Phase Space', color='purple')
plt.xlabel('B (Species)')
plt.ylabel('S (Resource)')
plt.title('Phase Diagram with Time-Dependent r_B and r_S')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()