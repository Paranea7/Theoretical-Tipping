import numpy as np
# 设置随机种子，以便结果可重复
np.random.seed(0)

# 生成随机噪声序列
noise = np.random.normal(0, 1, 1000)  # 生成100个符合正态分布(0, 1)的随机数

# 将随机噪声序列写入文件
with open('random_noise.txt', 'w') as f:
    for value in noise:
        f.write(f'{value}\n')

print(f'Random noise sequence saved to random_noise.txt')