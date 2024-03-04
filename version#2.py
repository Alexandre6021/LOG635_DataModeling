from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
import pickle

class NeuralNetwork():
    def __init__(self, nb_inputs, nb_outputs, nb_nodes_per_layer, learning_rate):
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs
        self.nb_nodes_per_layer = nb_nodes_per_layer
        self.learning_rate = learning_rate
        self.loss = []
        self.A = np.zeros(3, dtype=object)
        self.Z = np.zeros(3, dtype=object)
        self.W = np.zeros(3, dtype=object)
        self.B = np.zeros(3, dtype=object)
        
        # Weight and Bias
        self.init_weights_bias()
        
    
    def init_weights_bias(self):
        # Weights
        #print("LE NOMBRE DE IMPUT EST DE ", self.nb_inputs)
        self.W[0] = np.random.randn(self.nb_inputs, self.nb_nodes_per_layer)
        self.W[2] = np.random.randn(self.nb_nodes_per_layer, self.nb_outputs)
            
        # Bias
        self.B[0] = np.ones((self.nb_nodes_per_layer))
        self.B[2] = np.ones(self.nb_outputs)
        
        # Fill weight and bias for hidden layers
        self.W[1] = np.random.randn(self.nb_nodes_per_layer, self.nb_nodes_per_layer)
        self.B[1] = np.ones((self.nb_nodes_per_layer))
         
        
    def sigmoid(self, z):
        s = 1 / (1 + np.exp(-z))
        return s
    
    def derived_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def relu(self, z):
        s = np.maximum(0,z)
        return s
    
    def derived_relu(self, z):
        zprime = z
        zprime[zprime<=0] = 0
        zprime[zprime>0] = 1
        return zprime
    
    def softmax(self, z):
        s = (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T
        return s

    def derived_softmax(self, z):
        s = self.softmax(z)
        return s * (np.eye(s.size) - s.reshape(-1, 1))
    
    def entropy_loss(self, y, y_pred):
        eps = np.finfo(float).eps
        return -np.sum(y * np.log(y_pred + eps))
    
    def forward(self, X):
        # Forward propagation through our network
        # input to hidden
        self.A0 = X
        #print(f"A0 = {self.A0.shape} x {self.W[0].shape} + {self.B[0].shape}")

        self.Z[0] = np.dot(self.A0, self.W[0]) + self.B[0]
        self.A[0] = self.relu(self.Z[0])
        
        # print(f"Layer0 = {self.A0.shape} x {self.Z[0].shape}")
        
        #traverse hidden
        self.Z[1] = np.dot(self.A[0], self.W[1]) + self.B[1]
        self.A[1] = self.relu(self.Z[1])
        #print(f"Layer{i} = {self.A[i-1].shape} x {self.Z[i].shape}")
        
        # hidden to output
        self.Z[2] = np.dot(self.A[1], self.W[2]) + self.B[2]
        self.A[2] = self.softmax(self.Z[2])
        #print(f"Layer{self.nb_hidden_layers} = {self.A[self.nb_hidden_layers-1].shape} x {self.Z[self.nb_hidden_layers].shape}")
        
        #print(self.A[self.nb_hidden_layers])
        return self.A[2]
    
    def backward(self, X, y):
        m = X.shape[0]  # number of examples
        
        dA = np.zeros(2, dtype=object)
        dZ = np.zeros(3, dtype=object)
        dW = np.zeros(3, dtype=object)
        db = np.zeros(3, dtype=object)
        
        #Output layers
        # Error in output
        dZ[2] = self.A[2]-y
        # Delta for the weights w2
        dW[2] = (1./m) * np.dot(self.A[1].T, dZ[2])
        # Delta for the bias b2
        db[2] = np.sum(dZ[2], axis=0)  # sum across columns
        # Weights/bias update
        self.W[2] -= self.learning_rate * dW[2]
        self.B[2] -= self.learning_rate * db[2]
        
        #Hidden layers
        dA[1] = np.dot(dZ[2], self.W[2].T)

        dZ[1] = self.A[1] * self.derived_relu(self.Z[1])
        # Delta for the weights wn
        dW[1] = (1./m) * np.dot(self.A[1].T, dZ[1])
        # Delta for the bias b2
        db[1] = np.sum(dZ[1], axis=0)  # sum across columns
        # Update weights/bias
        self.W[1] -= self.learning_rate * dW[1]
        self.B[1] -= self.learning_rate * db[1]

        #Input layers
        dA[0] = np.dot(dZ[1], self.W[1].T)
        dZ[0] = dA[0] * self.derived_relu(self.Z[0])
        # Delta for the weights w1
        dW[0] = (1./m) * np.dot(X.T, dZ[1])
        # Delta for the bias b1
        db[0] = (1./m) * np.sum(dZ[1], axis=0)  # sum across columns

        # Wights and biases update
        self.W[0] -= self.learning_rate * dW[0]
        self.B[0] -= self.learning_rate * db[0]


    def train(self, X, y, nb_iterations):
        for i in range(nb_iterations):
            y_pred = self.forward(X)
            loss = self.entropy_loss(y, y_pred)
            self.loss.append(loss)
            self.backward(X, y)

            # why we start i at 1 ?
            if i == 0 or i == nb_iterations-1:
                print(f"Iteration: {i+1}")
                print(tabulate(zip(X, y, [np.round(y_pred) for y_pred in self.A[self.nb_hidden_layers]] ), headers=["Input", "Actual", "Predicted"]))
                print(f"Loss: {loss}")                
                print("\n")

    def predict(self, X):
        return np.round(self.forward(X))