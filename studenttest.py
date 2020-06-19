from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
import studNet


data = datasets.make_moons(n_samples=2500, noise=0.1)

X = data[0]
y = np.expand_dims(data[1], 1)

cond = ['red' if value == 1 else 'blue' for value in y]
plt.scatter(X[:,0], X[:,1], s=1, c=cond)
plt.title('Scatter plot for input data')
plt.show()


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, )
xPass = x_train.T
yPass = y_train.T

#test_all = studNet.test_train_all(xPass, yPass, x_test.T, y_test.T)
studentNet = studNet.test_train(xPass, yPass, x_test.T, y_test.T)

results = studentNet.predict(x_test.T)

print('Accuracy (noise = 0.1) = ', accuracy_score(y_test.T[0], results[0]))



