import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

def sigmoid(z):
    '''
    Sigmoid function on a vector this is included for use as your activation function
    
    Input: a vector of elements to preform sigmoid on
    
    Output: a vector of elements with sigmoid preformed on them
    '''
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def reludriv(z):
    return np.where(z > 0, 1, 0)

def sigmoidderiv(z):
    '''
    The derivative of Sigmoid, you will need this to preform back prop
    '''
    return sigmoid(z) * (1 - sigmoid(z))

class NeuralNetwork(object):     
    '''
    This Object outlines a basic neuralnetwork and the methods that it will need to function
    
    We have included an init method with a size parameter:
        Size: A 1D array indicating the node size of each layer
            E.G. Size = [2, 4, 1] Will instantiate weights and biases for a network
            with 2 input nodes, 1 hidden layer with 4 nodes, and an output layer with 1 node
        
        test_train defines the sizes of the input and output layers, but the rest is up to your implementation
    
    In this network for simplicity all nodes in a layer are connected to all nodes in the next layer, and the weights and
    biases and intialized as such. E.G. In a [2, 4, 1] network each of the 4 nodes in the inner layer will have 2 weight values
    and one biases value.
    
    '''

    def __init__(self, size, seed=42):
        '''
        Here the weights and biases specified above will be instantiated to random values
        Your network will change these values to fit a certain dataset by training
        '''

        self.seed = seed
        np.random.seed(self.seed)
        self.size = size
        self.weights = [np.random.randn(self.size[i], self.size[i-1]) * np.sqrt(1 / self.size[i-1]) for i in range(1, len(self.size))]
        self.biases = [np.random.rand(n, 1) for n in self.size[1:]]

    def cross_entropy(self, a, y):
        #res1 = np.array([1e-15 if value == 0 else value for value in a[0]])
        #res2 = np.array([1e-15 if value == 0 else value for value in (1-a)[0]])
        return np.sum(-(np.log(a).dot(y.T) + np.log(1-a).dot((1-y).T)))

    def reluderiv(self, z1):
        return np.where(z1 > 0, 1, 0)


    def train(self, X_tp, y_tp, num_epoch, batch_size, eta, num_hidden_layer, x_test_tp, y_test_tp):
        num_data = X_tp.shape[1]
        num_batch = int(num_data / batch_size)

        max_accuracy = 0

        index_arr = np.arange(0, num_data)
        loss_lst = []
        num_epoch_copy = num_epoch

        if num_hidden_layer == 1:
            while num_epoch > 0:
                num_epoch = num_epoch - 1
                loss = 0

                np.random.shuffle(index_arr)

                for i in range(num_batch):
                    w1 = self.weights[0]
                    w2 = self.weights[1]
                    b1 = self.biases[0]
                    b2 = self.biases[1]

                    random_index = index_arr[batch_size * i : batch_size * (i+1)]

                    x_batch = X_tp.T[random_index].T
                    y_batch = y_tp.T[random_index].T

                    # feed forward
                    z1 = np.dot(w1, x_batch) + b1
                    r = relu(z1)
                    z2 = np.dot(w2, r) + b2
                    output = sigmoid(z2)

                    results = self.predict(x_test_tp)
                    accuracy = accuracy_score(y_test_tp[0], results[0])
                    max_accuracy = np.maximum(max_accuracy, accuracy)
                    loss = loss + self.cross_entropy(output, y_batch)

                    dLdw2 = np.dot(output - y_batch, r.T) / batch_size
                    dLdw1 = np.dot(w2.T, output - y_batch).dot(x_batch.T) / batch_size
                    dLdb1 = np.expand_dims(np.mean(np.dot(w2.T, output - y_batch), axis=1), axis=1)
                    dLdb2 = np.mean(output - y_batch, axis=1)

                    self.weights[1] = w2 - eta * dLdw2
                    self.weights[0] = w1 - eta * dLdw1
                    self.biases[1] = b2 - eta * dLdb2
                    self.biases[0] = b1 - eta * dLdb1


                # average loss for each epoch
                avg_loss = loss / num_batch
                #print(avg_loss)
                loss_lst.append(avg_loss)

            print('max accuracy = ', max_accuracy)
            print('final loss = ', loss_lst[len(loss_lst) - 1])
        # num_hidden_layer == 2
        else:
            while num_epoch > 0:
                num_epoch = num_epoch - 1
                loss = 0

                random.shuffle(index_arr)

                for i in range(num_batch):
                    w1 = self.weights[0]
                    w2 = self.weights[1]
                    w3 = self.weights[2]
                    b1 = self.biases[0]
                    b2 = self.biases[1]
                    b3 = self.biases[2]

                    random_index = index_arr[batch_size * i: batch_size * (i + 1)]

                    x_batch = X_tp.T[random_index].T
                    y_batch = y_tp.T[random_index].T

                    # feed forward for 2 hidden layers
                    z1 = np.dot(w1, x_batch) + b1
                    r1 = relu(z1)
                    z2 = np.dot(w2, r1) + b2
                    r2 = relu(z2)
                    z3 = np.dot(w3, r2) + b3
                    a = sigmoid(z3)

                    loss = loss + self.cross_entropy(a, y_batch)
                    results = self.predict(x_test_tp)
                    accuracy = accuracy_score(y_test_tp[0], results[0])
                    max_accuracy = np.maximum(max_accuracy, accuracy)

                    self.weights[0] = w1 - eta * w2.T.dot(np.dot(w3.T, a-y_batch)).dot(x_batch.T) / batch_size
                    self.biases[0] = b1 - eta * np.expand_dims(np.mean(w2.T.dot(np.dot(w3.T, a-y_batch)), axis=1), axis=1)
                    self.weights[1] = w2 - eta * w3.T.dot(a-y_batch).dot(r1.T) / batch_size
                    self.biases[1] = b2 - eta * np.expand_dims(np.mean(w3.T.dot(a-y_batch), axis=1), axis=1)
                    self.weights[2] = w3 - eta * np.dot(a-y_batch, r2.T) / batch_size
                    self.biases[2] = b3 - eta * np.mean(a-y_batch, axis=1)


                # average loss for each epoch
                avg_loss = loss / batch_size
                #print(avg_loss)
                loss_lst.append(avg_loss)

            print('max accuracy = ', max_accuracy)
        # plotting the loss

        x = np.arange(num_epoch_copy)
        y = np.array(loss_lst)
        plt.scatter(x, y, c='red', s=3)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()

    def predict(self, a):
        '''
       Input: a: list of list of input vectors to be tested
       
       This method will test a vector of input parameter vectors of the same form as X in test_train
       and return the results (Zero or One) that your trained network came up with for every element.
       
       This method does this the same way the included forward method moves an input through the network
       but without storying the previous values (which forward stores for use with the delta function you must write)
        '''
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = sigmoid(z)
        predictions = (a > 0.5).astype(int)
        return predictions

def test_train(X_tp, y_tp, x_test_tp, y_test_tp):
    inputSize = np.size(X_tp, 0)

    num_data = X_tp.shape[1]
    print('number of input data = ', num_data)
    num_epoch = 1000
    batch_size = 50
    eta = 0.0001
    num_hidden_nodes = 30
    num_hidden_nodes2 = 30

    print('Training...[epoch = %d, batch_size = %d, eta = %f, hidden_nodes = %d, hidden_nodes = %d]' %(num_epoch, batch_size, eta, num_hidden_nodes, num_hidden_nodes2))
    retNN = NeuralNetwork([inputSize, num_hidden_nodes, num_hidden_nodes2, 1])

    retNN.train(X_tp, y_tp, num_epoch, batch_size, eta, 2, x_test_tp, y_test_tp)

    return retNN
