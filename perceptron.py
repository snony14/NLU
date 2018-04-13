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
Y = np.array([0,0,0,1])

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

def sigmoid_dev(x):
    return x*(1.0-x)

def feedforward(X, weights, b):
    #print(X.shape, weights.shape)
    pred_y = X.dot(weights) + b
    #print(sigmoid_dev(pred_y))
    return sigmoid(pred_y)


def train(X, Y, weights,b):
    epochs = 100000
    W = weights
    pred = feedforward(X, W,b)
    loss = (-Y*np.log(pred)).sum()
    #print(W.shape)
    v = np.array([0,0])
    beta = 0
    for f in range(epochs):
        for i in range(len(X)):
            pred_i = feedforward(X[i], W, b)
            #print(Y[i], pred_i)
            param_grad = (Y[i]-pred_i)*sigmoid_dev(pred_i)
            v = v + param_grad*X[i]
            beta = beta + param_grad
        W = W - 0.1*v/4
        b = b -0.1*beta/4
        pred = feedforward(X, W,b)
        loss = (-Y*np.log(pred)).sum()
        v *=0
        beta *=0
        #print(loss)
    return W

#print(weights)
weights = train(X, Y, weights,b)
#print(weights)
print(feedforward(X, weights,b))

# for i in range(epochs):
#     v = 0
#     for j in range(len(X)):
#         pred_j = feedforward(X[j], W)
#         #print(pred_j)
#         v += Y[j]*(1-pred_j)*X[j]
#     #print(v)
#     W = W - 0.01*v
