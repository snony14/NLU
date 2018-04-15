import numpy as np

np.random.seed(20130625)
X_lst = [np.random.rand(2) for i in range(6)]

V = np.random.rand(2,2)
U = np.random.rand(2,2)
W = np.random.rand(10,2)
init =np.random.rand(2)
print("V matrix:\n",V)
print("U matrix:\n",U)
print("W matrix:\n",W)

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

def softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)


#implementation of rnn functions

def forwardRNN(X, V, U,W, init):
    '''
    X: is a list of input vector each of size D
    V: is the matrix of size R x D, maps the input vector into a vector of size R
    U: is the matrix of size R x D, maps the representation vector until time t into a vector of size R
    W: is the matrix of size V x R, maps the sum of representation to a vector of size V
    '''
    hidden_states = []
    outputs = []
    h_t_1 = init
    hidden_states.append(h_t_1)
    i = 0
    for x in X:
        input_mapped = V.dot(x)
        #print(i,U.dot(h_t_1).shape, input_mapped.shape)
        hidden_mapped = input_mapped + U.dot(h_t_1)
        h_t = sigmoid(hidden_mapped)
        h_t_1 = h_t
        hidden_states.append(h_t_1)
        output_mapped = W.dot(h_t)
        o_t = softmax(output_mapped)
        outputs.append(o_t)
        i += 1
    return (hidden_states, outputs)

def bppt(X,V, U, W, init):
    pass

h_states, o_states = forwardRNN(X_lst, V, U, W, init)
