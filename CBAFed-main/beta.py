import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

# 假设您有1000个0到1之间的概率值，将其存储在一个NumPy数组中
data = np.random.rand(1000)  # 这里使用随机生成的示例数据

# 使用最大似然估计估计Beta分布的参数
alpha_hat = np.mean(data) * ((np.mean(data) * (1 - np.mean(data)) / np.var(data)) - 1)
beta_hat = (1 - np.mean(data)) * ((np.mean(data) * (1 - np.mean(data)) / np.var(data)) - 1)
from scipy import stats
# 绘制拟合的Beta分布曲线
x = np.linspace(0, 1, 1000)  # 生成用于绘制曲线的x值
pdf = beta.pdf(x, alpha_hat, beta_hat)  # 计算概率密度函数
mean1 = stats.beta(alpha_hat,beta_hat).mean()
mean2 = np.mean(data)
plt.plot(x, pdf, label='Fitted Beta PDF', color='red')
plt.hist(data, bins=30, density=True, alpha=0.5, label='Histogram of Data')
plt.legend()
plt.xlabel('Probability Values')
plt.ylabel('Probability Density')
plt.title('Fitting a Beta Distribution to Data')
plt.show()

print(f"Estimated Alpha: {alpha_hat}")
print(f"Estimated Beta: {beta_hat}")
