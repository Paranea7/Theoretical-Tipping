import numpy as np
import matplotlib.pyplot as plt

# 生成基础脉冲周期
period = 30
pulse_base = [0.1 if 4 <= i <= 27 else 1 for i in range(1, period + 1)]

# 生成连续时间序列（5个周期为例）
num_cycles = 5
continuous_signal = np.tile(pulse_base, num_cycles)

# 添加高斯噪声
noise_strength = 0.05  # 噪声强度（可调参数）
noisy_signal = continuous_signal + np.random.normal(0, noise_strength, len(continuous_signal))

# 绘图
plt.figure(figsize=(15, 5))
plt.plot(continuous_signal, 'b-', linewidth=2, alpha=0.7, label='original signal')
plt.plot(noisy_signal, 'r-', alpha=0.5, label='noise')

plt.xlabel('time')
plt.ylabel('Force')
plt.legend()
plt.grid(True)
plt.xlim(0, len(continuous_signal))

# 标注周期分隔线
for i in range(num_cycles):
    plt.axvline(i * period, color='gray', linestyle='--', alpha=0.5)

plt.show()