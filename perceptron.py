import numpy as np
#import math

np.random.seed(20130625)
weights = np.random.rand(2,2)
weight2 = np.random.rand(1,2)
print(weight2)
b1 = np.random.rand(2)
b2 = np.random.rand(1)
#print(weights)

X = np.array([[0,0],
[0,1],
[1,0],
[1,1]
])
# Y = np.array([[0,0],
# [0,1],
# [1,0],
# [1,1]
# ])
def getLogicFunc(X,ft = 0):
    Y = []
    func = None
    if ft == 0:
        #and function
        func = np.logical_and
    elif ft == 1:
        #logical or
        func = np.logical_or
    elif ft == 2:
        #logical XOR
        func = np.logical_xor
    else:
        func = np.logical_and

    Y = [[func(x[0],x[1])] for x in X]
    return np.array(Y)

#print(X.shape)
Y = getLogicFunc(X,2)*1#np.array([1,1,1,0])
# print(Y)
def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    return (x > 0)*1

def sigmoid_dev(x):
    return x*(1.0-x)

def feedforward(X, weights, b):
    #print(X.shape, weights.shape)
    pred_y = weights.dot(X.T).T + b#X.dot(weights.T) + b
    #print(sigmoid_dev(pred_y))
    return sigmoid(pred_y)


def train(X, Y, weights,b):
    epochs = 15000
    W = weights
    beta = b
    for i in range(epochs):
        pred = feedforward(X, W,beta)
        loss = Y - pred
        loss_delta = loss * sigmoid_dev(pred)
        print(X.shape,loss_delta.shape, "loss shape")
        beta += loss_delta.sum()
        W += np.dot(X.T, loss_delta)
        #break
    return (W, beta)

#Learns a more complicated function
def train_multi(X, Y, W1, W2, b1, b2):
    epochs = 1500
    for i in range(epochs):
        Y1 = feedforward(X, W1, b1)
        #print(Y1.shape)
        Y2 = feedforward(Y1, W2, b2)
        #print(Y2.shape, "lala")
        #error with respect to Y2:
        loss_y2 = Y - Y2#4
        #The gradient of loss with respect to this input
        loss_y2_delta = loss_y2 * sigmoid_dev(Y2)
        #find the gradient w.r.t inputs
        loss_delta_y1 = loss_y2_delta.dot(W2)
        W2 +=  np.dot(loss_y2_delta.T, Y1)
        b2 += np.sum(loss_y2_delta, axis=0)
        #print(loss_delta_y1.shape)
        W1 += (loss_delta_y1*sigmoid_dev(Y1)).T.dot(X)
        b1 += np.sum((loss_delta_y1*sigmoid_dev(Y1)), axis=0)
        #print(W1.shape)
        #break
    return (W1, W2, b1, b2)


#print(weights)
# weights, b = train(X, Y, weights,b)
W1, W2, b1, b2 = train_multi(X, Y, weights, weight2, b1, b2)
x =np.array([[1,0]])
out1 = feedforward(X,W1,b1)
print(out1)
out2 = feedforward(out1, W2,b2)
print(out2)
#print((feedforward(x, weights,b)>0.5)*1)
