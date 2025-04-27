import numpy as np

step_function = lambda x: 1 if x >= 0 else 0

training_data = [
    {'input': [int(b) for b in f"{n:06b}"], 'label': (n % 2) ^ 1}
    for n in range(48, 58)
]

weights = np.array([0, 0, 0, 0, 0, 1])

for data in training_data:
    inpt = np.array(data['input'])
    label = data['label']
    output = step_function(np.dot(inpt, weights))
    error = label - output
    weights += inpt * error


# Take Input from user
j = int(input("Enter a Number (0-63): "))

inpt = np.array([int(x) for x in list('{0:06b}'.format(j))])
output = "odd" if step_function(np.dot(inpt, weights)) == 0 else "even"
print(j, " is ", output)
