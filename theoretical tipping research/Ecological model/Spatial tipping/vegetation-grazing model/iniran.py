import numpy as np

# 随机初始化V
V = np.random.uniform(1.5, 3.5, (50, 50))

# 保存V到CSV文件
np.savetxt('1.5-3.5.csv', V, delimiter=',')