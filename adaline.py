import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import os
class adaline:
    """
        Adaline Network that takes certain learning rate, input, outputs, and random number
        to get certain accuracy on predicting output.
        learning_rate: learning rate of the network
        inputs: training input for the network without it's respective output. Generally
        determined by 'X'
        output: the result input should give. Generally determined by 'y'
        random: option to put random number or as per assignment default 'False'
        i.e. uses assignment numbers.
        seed: seed for random number. Default it is `42`
    """
    def __init__(self, learning_rate, inputs, outputs, random = False, seed = 42):
        np.random.seed(seed)
        if random:
            self.weights = np.random.uniform(0, 1, size = (2,))
            self.biases = np.random.uniform(0, 1)
        else:
            self.weights = np.array([0.2, 0.2])
            self.biases = 0.2
        self.learning_rate = learning_rate
        self.combined = list(zip(inputs, outputs))

    def tuning_weights(self):
        """
            For Tuning weights on the basis of adaline's algorithm
        """
        self.collection_of_weights = []
        self.collection_of_biases = []
        self.collection_of_errors = []
        self.collection_of_MSE = []
        self.collection_of_weights.append(self.weights)
        self.collection_of_biases.append(self.biases)
        for i in range(0, 6):
            self.MSE = []    # To collect the error whose when calculated mean of gives Mean Squared Error
            for inputs, outputs in self.combined:
                net = self.biases + np.dot(self.weights, inputs)
                if outputs - net == 0:
                    break
                self.weights = self.weights + np.dot(self.learning_rate*(outputs - net), inputs)
                self.biases = self.biases + self.learning_rate * (outputs - net)
                errors = (outputs - net)**2
                self.MSE.append(errors)
                self.collection_of_errors.append(errors)
                self.collection_of_weights.append(self.weights)
                self.collection_of_biases.append(self.biases)
            self.collection_of_MSE.append(np.mean(self.MSE))
        self.collection_of_errors.append("NaN")
        self.total_number_of_iteration = [x for x in range(1, len(self.collection_of_errors)+1)]
        self.total_number_of_Epoch = [x for x in range(1, len(self.collection_of_MSE)+1)]
    def output_csv_error(self, display = False, file_save = True):
        """
            Creates a dataframe of Number of iteration, Weights, Biases and errors.
            display: displays the result on the screen. Default = False.
            file_save: Saving to a file. Default = True
        """
        for_Error = {"Iteration number": self.total_number_of_iteration, "Weight": self.collection_of_weights, "Biases": self.collection_of_biases, "Errors": self.collection_of_errors}
        df_Error = pd.DataFrame(for_Error)
        if file_save:
            file_name = "Error" + str(datetime.datetime.now().strftime("%Y-%m-%d")) + str(datetime.datetime.now().strftime("%I-%M-%S %p"))+".csv"
            df_Error.to_csv(file_name, index = False)
        if display:
            print(df_Error)
    def output_csv_MSE(self, display = False, file_save = True):
        """
            Creates a dataframe of number of epoch and MSE.
            display: displays the result on the screen. Default = False.
            file_save: Saving to a file. Default = True
        """
        for_MSE = {"Iteration number": self.total_number_of_Epoch, "Errors": self.collection_of_MSE}
        df_MSE = pd.DataFrame(for_MSE)
        if file_save:
            file_name = "Error-MSE" + str(datetime.datetime.now().strftime("%Y-%m-%d")) + str(datetime.datetime.now().strftime("%I-%M-%S %p"))+".csv"
            df_MSE.to_csv(file_name, index = False)
        if display:
            print(df_MSE)
    def display_graph(self, save_graph = True):
        """
            Displays graph of Iteration vs Errors and
            Epoch vs MSE.
            save_graph: option to save the graph. Default = True
        """
        fig, axs = plt.subplots(2, figsize = (5,8))
        fig.suptitle("Graph to show changes in errors with iterations or epoch")
        axs[0].plot(self.total_number_of_iteration[:-1], self.collection_of_errors[:-1])
        axs[0].set(xlabel ="Total Number of iteration", ylabel = "Errors")
        axs[1].plot(self.total_number_of_Epoch, self.collection_of_MSE)
        axs[1].set(xlabel ="Total Number of Epoch", ylabel = "Mean Squared Error")
        if save_graph:
            fig_name = str(datetime.datetime.now().strftime("%Y-%m-%d")) + str(datetime.datetime.now().strftime("%I-%M-%S %p"))+".png"
            fig.savefig(os.path.abspath(os.getcwd())+"\\"+fig_name)
        plt.show()

inputs = [[1,1],[1,-1],[-1,1],[-1,-1]]
outputs = [-1, 1, -1, -1]
learning_rate = 0.2

training = adaline(learning_rate, inputs, outputs)
training.tuning_weights()
training.display_graph()
training.output_csv_error()
training.output_csv_MSE()