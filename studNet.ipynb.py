import numpy as np

def sigmoid(z):
    '''
    Sigmoid function on a vector this is included for use as your activation function
    
    Input: a vector of elements to preform sigmoid on
    
    Output: a vector of elements with sigmoid preformed on them
    '''
    return 1 / (1 + np.exp(-z))


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

        print('The shape of initial weights')
        print(self.weights[0])
        print(self.weights[1])
        print('The shape of initial biases')
        print(len(self.biases))
        
    def forward(self, input):
        '''
        Perform a feed forward computation 
        Parameters:

        input: data to be fed to the network with (shape described in spec)

        returns:

        the output value(s) of each example as ‘a’

        The values before activation was applied after the input was weighted as ‘pre_activations’

        The values after activation for all layers as ‘activations’

        You will need ‘pre_activaitons’ and ‘activations’ for updating the network
        '''
        print('feed forward function')
        print('input =', input)
        a = input
        pre_activations = []
        activations = [a]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = sigmoid(z)
            pre_activations.append(z)
            activations.append(a)
        print('done with forward function')
        print(a)
        return a, pre_activations, activations

    def backpropagate(self, a, pre_activations, activations):
        print('backpropagation function')

    def train(self, X, y):
        a, pre_activations, activations = self.forward(X)
        print('RESULT for forward propagation....')
        print('a')
        print(a)
        print('pre_activations')
        print(pre_activations)
        print('activations')
        print(activations)


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
    
'''
This is the function that we will call to test your network.

It must instantiate a network, which we include.

It must then train the network given the passed data, where x is the parameters in form:
        [[1rst parameter], [2nd parameter], [nth parameter]]
    
        Where if there are 100 training examples each of the n lists inside the list above will have 100 elements
        
    Y is the target which is guarenteed to be binary, or in other words true or false:
    Y will be of the form: 
        [[1, 0, 0, ...., 1, 0, 1]]
        
        (where 1 indicates true and zero indicates false)

'''
def test_train(X, y):
    print('test_train function')
    inputSize = np.size(X, 0)
    print('input size = ', inputSize)
    
    #feel free to change the inside (hidden) layers to best suite your implementation
    #but the sizes of the input layer and output layer (inputSize and 1) must NOT CHANGE
    retNN = NeuralNetwork([inputSize, 4, 1])
    #train your network here
    retNN.train(X, y)
    
    #then the function MUST return your TRAINED nueral network
    return retNN


X = np.array([[0,0],
              [1,0],
              [5,1],
              [5,2],
              [5,5],
              [5,6],
              [0,5],
              [0,4]])

y = [0,0,1,1,0,0,1,1]

test_train(X, y)