import numpy as np
#import math

np.random.seed(20130625)
weights = np.random.rand(2)
b = np.random.rand(1)
#print(weights)

X = np.array([[0,0],
[0,1],
[1,0],
[1,1]
])

#print(X.shape)
Y = np.array([0,1,1,1])

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    return (x > 0)*1

def sigmoid_dev(x):
    return x*(1.0-x)

def feedforward(X, weights, b):
    #print(X.shape, weights.shape)
    pred_y = X.dot(weights) + b
    #print(sigmoid_dev(pred_y))
    return sigmoid(pred_y)


def train(X, Y, weights,b):
    epochs = 15000
    W = weights
    #print(W.shape)
    v = np.array([0,0])
    beta = b
    for i in range(epochs):
        pred = feedforward(X, W,beta)
        loss = Y - pred
        loss_delta = loss * sigmoid_dev(pred)
        beta += loss_delta.sum()
        W += np.dot(X.T, loss_delta)
    return (W, beta)

#print(weights)
weights, b = train(X, Y, weights,b)
x =np.array([[1,1]])
print(weights, b)
print((feedforward(x, weights,b)>0.5)*1)
