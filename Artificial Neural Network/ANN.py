import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

dig = load_digits()
plt.matshow(dig.images[0])
plt.show()

print("Resolution of images: ", len(dig.images[0]), "x", len(dig.images[0][0]))

onehot_target = pd.get_dummies(dig.target)
print(onehot_target.iloc[0, :])

x_train, x_val, y_train, y_val = train_test_split(dig.data, onehot_target, test_size=0.1, random_state=20)
# shape of x_train :(1617, 64)
# shape of y_train :(1617, 10)


# Model

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def softmax(s):
    # for numerical stability, values are normalised
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

def error(pred, real):
    # n_samples = real.shape[0] # 1617 observations during training
    # logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    # loss = np.sum(logp)/n_samples
    # return loss
    loss = - np.mean(np.multiply(real, np.log(pred)))
    return loss

def cross_entropy_and_softmax_derv(pred, real):
    # combined derivative of softmax and cross entropy using chain rule
    n_samples = real.shape[0]
    res = pred - real
    return res/n_samples

def sigmoid_derv(x):
    return sigmoid(x) * (1 - sigmoid(x))


class ANN:
    def __init__(self, x, y):
        self.x = x
        neurons = 128 # neurons for hidden layers
        self.lr = 0.5
        ip_dim = x.shape[1] # input layer size 64
        op_dim = y.shape[1] # output layer size 10

        self.w1 = np.random.randn(ip_dim, neurons) # weights
        self.b1 = np.zeros((1, neurons)) # biases
        self.w2 = np.random.randn(neurons, neurons)
        self.b2 = np.zeros((1, neurons))
        self.w3 = np.random.randn(neurons, op_dim)
        self.b3 = np.zeros((1, op_dim))
        # weights will be given random values which are close to 0 but not 0
        self.y = y

    def feedforward(self):
        self.z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = softmax(self.z3)

    def backprop(self):
        # we use cross entropy loss because cross-entropy function is able to compute error between two probability distributions.
        loss = error(self.a3, self.y)
        print(' Loss :', loss)

        a3_delta = cross_entropy_and_softmax_derv(self.a3, self.y)
        # z3_delta step is skipped because some terms get cancelled when we take derivative of cross entry and then softmax
        # therefore we combine the two steps into one and calculate a3_delta directly
        z2_delta = np.dot(a3_delta, self.w3.T)
        a2_delta = z2_delta * sigmoid_derv(self.z2)
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * sigmoid_derv(self.z1)
        # z(i)_delta is basically dL/da(i)
        # a(i)_delta is basically dL/dz(i)
        # L is cross entropy loss here

        self.w3 -= self.lr * np.dot(self.a2.T, a3_delta)
        self.b3 -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
        self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
        self.b2 -= self.lr * np.sum(a2_delta, axis=0)
        self.w1 -= self.lr * np.dot(self.x.T, a1_delta)
        self.b1 -= self.lr * np.sum(a1_delta, axis=0)

    def predict(self, data):
        self.x = data
        self.feedforward()
        return self.a3.argmax()


model = ANN(x_train/16.0, np.array(y_train))

epochs = 1000
for i in range(epochs):
    print("Epochs :", i+1, end="")
    model.feedforward()
    model.backprop()


def get_acc(x, y):
    acc = 0
    for xx,yy in zip(x, y):
        s = model.predict(xx)
        if s == np.argmax(yy):
            acc +=1
    return acc/len(x)*100


print("Training accuracy : ", get_acc(x_train/16, np.array(y_train)))
print("Test accuracy : ", get_acc(x_val/16, np.array(y_val)))
