import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

X_train = np.array([
    [[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]],  # 0
    [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],  # 1
    [[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]],  # 2
    [[1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]]   # 3
])
y_train = [0, 1, 2, 3]

fig, axes = plt.subplots(1, len(X_train), figsize=(5, 3))
for i, ax in enumerate(axes):
    ax.imshow(X_train[i], cmap='gray_r')
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis("off")
plt.show()

clf = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', max_iter=1000)
clf.fit(X_train.reshape(len(X_train), -1), y_train)

X_test = np.array([
    [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],  # 1
    [[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]],  # 2
    [[1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],  # 3
    [[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]]   # 0
])

y_pred = clf.predict(X_test.reshape(len(X_test), -1))
for i, pred in enumerate(y_pred):
    print(f"Test {i + 1}: Predicted Number is {pred}")
