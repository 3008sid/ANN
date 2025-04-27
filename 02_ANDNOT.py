import numpy as np

def mp_neuron(inputs, weights, threshold):
    weighted_sum = np.dot(inputs, weights)
    output = int(weighted_sum >= threshold)
    return output
    
def and_not(x1, x2):
    weights = [1, -1] 
    threshold = 1   
    inputs = np.array([x1, x2])
    output = mp_neuron(inputs, weights, threshold)
    return output

result = [(i, j, and_not(i, j)) for i in range(2) for j in range(2)]

print(f"{'X1':<5} {'X2':<5} {'Y':<5}")
print("-" * 15)

for row in result:
    print(f"{row[0]:<5} {row[1]:<5} {row[2]:<5}")