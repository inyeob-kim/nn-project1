def backpropagate(self, a, pre_activations, activations, y):
    print('y shape =', y.shape)
    print('activations[2] shape = ', activations[2].shape)
    print('self.weights[1] shape = ', self.weights[1].shape)
    print('activations[1] shape = ', activations[1].shape)
    print('activations[0] shape = ', activations[0].shape)
    dLdw1 = -(y - activations[2]) * self.weights[1].dot(activations[1]) * (1 - activations[1]) * activations[0]
    dLdw2 = -(y - activations[2]) * activations[1]
    dLdb1 = -(y - activations[2]) * self.weights[1] * activations[1] * (1 - activations[1])
    dLdb2 = -(y - activations[2])

    g_weights = [dLdw1, dLdw2]
    g_biases = [dLdb1, dLdb2]

    return g_weights, g_biases


def update(self, g_weights, g_biases, eta):
    # updating
    for i in range(len(g_weights)):
        self.weights[i] = self.weights[i] - eta * g_weights[i]
        self.biases[i] = self.biases[i] - eta * g_biases[i]

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

        print('########### Result forward ###########')
        print('output')
        print(a)
        print('pre_activations, z')
        print(pre_activations)
        print('activations, a')
        print(activations)

        return a, pre_activations, activations




X = np.array([[0,0],
              [1,0],
              [5,1],
              [5,2],
              [5,5],
              [5,6],
              [0,5],
              [0,4],
              [1,5]])

y = np.array([0,0,1,1,0,0,1,1,1])

num_epoch = 10000

print('w1 shape = ', self.weights[0].shape)
print('w2 shape = ', self.weights[1].shape)
print('b1 shape = ', self.biases[0].shape)
print('b2 shape = ', self.biases[1].shape)
print('x_batch shape = ', x_batch.shape)
print('y_batch shape = ', y_batch.shape)
print('a shape = ', a.shape)
batch_size = 3
eta = 0.002
NN_model = test_train(X.T, y.T, num_epoch, batch_size, eta)

print('weight after update')
print('w1 shape = ', self.weights[0].shape)
print('x_batch shape = ', x_batch.shape)
print('b1 shape = ', b1.shape)
print('b2 shape - ', b2.shape)
print('w2 shape = ', w2.shape)
print('a shape = ', a.shape)
print('y_batch shape = ', y_batch.shape)
print('a-y_batch shape = ', (a - y_batch).shape)
print('r shape = ', r.shape)


data = datasets.make_moons(n_samples=2500, noise=0.2)


X = data[0]
y = np.expand_dims(data[1], 1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, )

xPass = x_train.T
yPass = y_train.T

studentNet = studNet.test_train(xPass, yPass)
results = studentNet.predict(x_test.T)
print('Accuracy')
print(accuracy_score(y_test.T[0], results[0]))


X = np.array([[0,0],
              [0,1],
              [1,0],
              [10,10],
              [9,10],
              [10,9],
              [0,10],
              [0,9],
              [0,11],
              [11,0],
              [10,0],
              [9,0]])
y = np.array([0,0,0,0,0,0,1,1,1,1,1,1])
y = y.reshape((len(y),1))

xPass = X.T
yPass = y.T

x_test = np.array([[1,1],
                   [10,0]])
y_test = np.array([0,1])
y_test = y_test.reshape((len(y_test), 1))

myNN = studNet.test_train(xPass, yPass)
results = myNN.predict(x_test.T)

print('Accuracy = ', accuracy_score(y_test.T[0], results[0]))



Training... [epoch = 250, batch_size = 50, eta = 0.0010, hidden_nodes = 20]
accuracy =  0.8733333333333333

Training... [epoch = 250, batch_size = 50, eta = 0.0050, hidden_nodes = 10]
accuracy =  0.88

Training... [epoch = 250, batch_size = 75, eta = 0.0010, hidden_nodes = 20]
accuracy =  0.872

Training... [epoch = 250, batch_size = 75, eta = 0.0050, hidden_nodes = 10]
accuracy =  0.8786666666666667

Training... [epoch = 300, batch_size = 25, eta = 0.0020, hidden_nodes = 10]
accuracy =  0.88

Training... [epoch = 300, batch_size = 50, eta = 0.0050, hidden_nodes = 10]
accuracy =  0.8826666666666667

num_epoch = 300
batch_size = 75
eta = 0.009
num_hidden_nodes = 10

loss = loss + self.cross_entropy(a2, y_batch)

reluderiv = np.where(z1, 1, 0)
print('reluderive shape = ', reluderiv.shape)
drdz1 = np.sum(reluderiv) / (reluderiv.shape[0] * reluderiv.shape[1])
print('w1 shape = ', self.weights[0].shape)
print('w2 shape = ', self.weights[1].shape)
print('b1 shape = ', self.biases[0].shape)
print('b2 shape = ', self.biases[1].shape)
print('x_batch shape = ', x_batch.shape)
print('y_batch shape = ', y_batch.shape)
print('a shape = ', a.shape)

self.weights[0] = w1 - eta * np.dot(w2.T, a - y_batch).dot(x_batch.T) * drdz1 / batch_size
self.biases[0] = b1 - eta * np.expand_dims(np.mean(np.dot(w2.T, a - y_batch) * drdz1, axis=1), axis=1)
self.weights[1] = w2 - eta * np.dot(a - y_batch, r.T) / batch_size
self.biases[1] = b2 - eta * np.mean(a - y_batch, axis=1)

print('Shape of w1 = ', w1.shape)
print('Shape of w2 = ', w2.shape)
print('Shape of b1 = ', b1.shape)
print('Shape of a1 = ', a1.shape)
print('Shape of a2 = ', a2.shape)
print('Shape of x = ', x_batch.shape)
print('Shape of y = ', y_batch.shape)

[epoch = 300, batch_size = 50, eta = 0.0090, hidden_nodes = 13, hidden_nodes = 0]

self.weights[0] = w1 - eta * w2.T.dot(a2 - y_batch).dot(a1.T).dot(1 - a1).dot(x_batch.T) / batch_size
self.weights[1] = w2 - eta * np.sum((a2 - y_batch).T.dot(w2), axis=0).reshape((1, w2.shape[1])) / batch_size
self.biases[0] = b1 - eta * np.sum(w2.T.dot(a2 - y_batch).dot(a1.T.dot(1 - a1)), axis=1).reshape(
    (b1.shape[0], 1)) / batch_size
self.biases[1] = b2 - eta * np.sum(a2 - y_batch) / batch_size

reluderiv = np.where(z1, 1, 0)
drdz1 = np.sum(reluderiv) / (reluderiv.shape[0] * reluderiv.shape[1])
self.weights[0] = w1 - eta * np.dot(w2.T, a - y_batch).dot(x_batch.T) * drdz1 / batch_size
self.biases[0] = b1 - eta * np.expand_dims(np.mean(np.dot(w2.T, a - y_batch) * drdz1, axis=1), axis=1)
self.weights[1] = w2 - eta * np.dot(a - y_batch, r.T) / batch_size
self.biases[1] = b2 - eta * np.mean(a - y_batch, axis=1)

self.weights[0] = w1 - eta * w2.T.dot(a - y_batch).dot(a.T).dot(1 - a).dot(x_batch.T) * self.reluderiv(z1) / batch_size
self.biases[0] = b1 - eta * np.expand_dims(
    np.mean(eta * w2.T.dot(a - y_batch).dot(a.T).dot(1 - a) * self.reluderiv(z1), axis=1), axis=1)
self.weights[1] = w2 - eta * (a - y_batch).dot(a.T).dot(1 - a).dot(r.T) / batch_size
self.biases[1] = b2 - eta * np.mean((a - y_batch).dot(a.T).dot(1 - a), axis=1)




data = datasets.make_moons(n_samples=2500, noise=0.2)



X = data[0]
y = np.expand_dims(data[1], 1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, )

xPass = x_train.T
yPass = y_train.T

studentNet = studNet.test_train(xPass, yPass, x_test.T, y_test.T)
results = studentNet.predict(x_test.T)
print('Accuracy (noise = 0.2) = ', accuracy_score(y_test.T[0], results[0]))