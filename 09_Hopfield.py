import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for p in patterns:
            p = p.reshape(-1, 1)
            self.weights += p @ p.T
        np.fill_diagonal(self.weights, 0)  # No self-connection

    def recall(self, pattern, steps=5):
        pattern = pattern.copy()
        for _ in range(steps):
            for i in range(self.size):
                raw = np.dot(self.weights[i], pattern)
                pattern[i] = 1 if raw >= 0 else -1
        return pattern

patterns = np.array([
    [1, -1, 1, -1, 1, -1],
    [-1, -1, 1, 1, -1, -1],
    [1, 1, -1, -1, 1, 1],
    [-1, 1, -1, 1, -1, 1]
])

hopfield = HopfieldNetwork(size=6)
hopfield.train(patterns)

test_input = np.array([1, 1, -1, -1, -1, 1])  # Slightly noisy version of third pattern
output = hopfield.recall(test_input)

print("Input pattern:  ", test_input)
print("Recalled pattern:", output)