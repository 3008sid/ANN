import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

sigmoid = lambda x: 1 / (1 + np.exp(-x))
tanh = lambda x: np.tanh(x)
softmax = lambda x: np.exp(x) / np.sum(np.exp(x))
relu = lambda x: np.maximum(0, x)
leaky_relu = lambda x: np.where(x > 0, x, 0.01 * x)
elu = lambda x, alpha=1.0: np.where(x > 0, x, alpha * (np.exp(x) - 1))

x = np.linspace(-5, 5, 500)

activation = [sigmoid, tanh, softmax, relu, leaky_relu, elu]
titles = ["Sigmoid", "Tanh", "Softmax", "ReLU", "Leaky ReLU", "ELU"]
colors = ["blue", "red", "green", "purple", "orange", "brown"]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for index, fun in enumerate(activation):
    y = fun(x)
    axes[index//3, index % 3].plot(x, y, color=colors[index])
    axes[index//3, index % 3].set_title(titles[index])
    axes[index//3, index % 3].grid(True)
    
for ax in axes.flat:
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")

plt.tight_layout()
plt.show()