import numpy as np

class ART1:
    def __init__(self, input_size, num_categories, vigilance):
        self.input_size = input_size
        self.num_categories = num_categories
        self.vigilance = vigilance

        # Initialize weight matrix (each category has its own weight vector)
        self.weights = np.ones((num_categories, input_size * 2))  # for complement coding

    def complement_code(self, input_vector):
        """Generate complement coded input vector"""
        return np.concatenate([input_vector, 1 - input_vector])

    def match_category(self, input_vector):
        """Compute normalized match scores"""
        scores = np.dot(self.weights, input_vector)
        norms = np.sum(input_vector)
        return scores / norms

    def train(self, inputs):
        for input_vector in inputs:
            input_vector = self.complement_code(input_vector)
            while True:
                match_scores = self.match_category(input_vector)
                chosen_category = np.argmax(match_scores)

                # Vigilance test
                min_sum = np.sum(np.minimum(input_vector, self.weights[chosen_category]))
                input_sum = np.sum(input_vector)

                if min_sum / input_sum >= self.vigilance:
                    # Update weights
                    self.weights[chosen_category] = np.minimum(input_vector, self.weights[chosen_category])
                    break
                else:
                    # Reset the category by setting its weight to zero
                    self.weights[chosen_category] = np.zeros_like(self.weights[chosen_category])

    def predict(self, input_vector):
        input_vector = self.complement_code(input_vector)
        match_scores = self.match_category(input_vector)
        return np.argmax(match_scores)

inputs = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
])

art = ART1(input_size=3, num_categories=4, vigilance=0.7)
art.train(inputs)

print("Predictions:")
for input_vector in inputs:
    category = art.predict(input_vector)
    print(f"Input: {input_vector}, Predicted Category: {category}")




