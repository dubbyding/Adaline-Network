import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

weights = np.random.uniform(0, 1, size = (2,))

biases = np.random.uniform(0, 1)

# weights = np.array([0.2, 0.2])
# biases = 0.2

learning_rate = 0.2

andnotInput = [[1,1],[1,-1],[-1,1],[-1,-1]]

outputs = [-1, 1, -1, -1]

combined = list(zip(andnotInput, outputs))

collection_of_weights = []

collection_of_biases = []

collection_of_errors = []

collection_of_MSE = []

for i in range(0,20):

    MSE = []

    for inputs, outputs in combined:
        net = biases + np.dot(weights, inputs)
        if outputs-net == 0:
            break
        collection_of_weights.append(weights)
        collection_of_biases.append(biases)
        weights = weights + np.dot(learning_rate*(outputs - net), inputs)
        biases = biases + learning_rate * (outputs - net)
        errors = (outputs - net)**2
        MSE.append(errors)
        collection_of_errors.append(errors)
    collection_of_MSE.append(np.mean(MSE))
    collection_of_weights.append(weights)
    collection_of_biases.append(biases)

total_number_of_iteration = [x for x in range(1, len(collection_of_errors)+1)]
total_number_of_Epoch = [x for x in range(1, len(collection_of_MSE)+1)]

fig, axs = plt.subplots(2, figsize = (5,8))
fig.suptitle("Graph to show changes in errors with iterations or epoch")
axs[0].plot(total_number_of_iteration, collection_of_errors)
axs[0].set(xlabel ="Total Number of iteration", ylabel = "Errors")
axs[1].plot(total_number_of_Epoch, collection_of_MSE)
axs[1].set(xlabel ="Total Number of Epoch", ylabel = "Mean Squared Error")
plt.show()
        