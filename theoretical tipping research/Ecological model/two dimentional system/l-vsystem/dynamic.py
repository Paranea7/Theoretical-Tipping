import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed

# 向量化的Lotka-Volterra模型
def lotka_volterra_vectorized(t, z, a, b, c, d):
    x = z[0]
    y = z[1]
    dxdt = a * x - b * x * y  # 猎物种群变化
    dydt = c * x * y - d * y  # 捕食者种群变化
    return np.array([dxdt, dydt])

# 定义一个函数以求解 ODE 和返回结果
def simulate_model(a, b, c, d, x0, y0, t_span):
    z0 = [x0, y0]
    solution = solve_ivp(lotka_volterra_vectorized, t_span, z0, args=(a, b, c, d), t_eval=np.linspace(t_span[0], t_span[1], 500))
    return solution.t, solution.y

# 设置模型参数
a = 1.0  # 猎物的增长率
b = 0.1  # 捕食率
c = 0.01  # 捕食者的效率
d = 1.0  # 捕食者的死亡率

# 初始条件
initial_conditions = [(40, 9), (30, 5), (50, 10), (60, 15)]  # 不同的初始条件
t_span = (0, 50)

# 并行计算多个初始条件下的动态变化
results = Parallel(n_jobs=-1)(delayed(simulate_model)(a, b, c, d, x0, y0, t_span) for x0, y0 in initial_conditions)

# 绘制结果
plt.figure(figsize=(12, 5))

# 动态变化
plt.subplot(1, 2, 1)
for t, y in results:
    plt.plot(t, y[0], label=f'Prey (x) initial ({y[0][0]}, {y[1][0]})')
    plt.plot(t, y[1], label=f'Predator (y) initial ({y[0][0]}, {y[1][0]})', linestyle='--')

plt.title('Prey-Predator Dynamics with Initial Conditions')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()

# 相轨迹
plt.subplot(1, 2, 2)
for _, y in results:
    plt.plot(y[0], y[1])  # 绘制相轨迹

plt.title('Phase Space')
plt.xlabel('Prey (x)')
plt.ylabel('Predator (y)')

plt.tight_layout()
plt.show()