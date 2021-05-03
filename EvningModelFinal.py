import pandas as pd
import random
import math
import statistics as st
import numpy as np
from tqdm import tqdm
import plotly.express as px

#Klasyfikator binarny

def sigmoid(x):
    if x < 0:
        return 1 - 1 / (1 + math.exp(x))
    else:
        return 1 / (1 + math.exp(-x))


def derivsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mse_loss(y, ypred):
    return st.mean([(y[x] - ypred[x]) ** 2 for x in range(len(y))])


def mse_derevative(y, ypred):
    return -2 * (y - ypred),


def linear(x):
    return x


def relu(x):
    return max(0, x)


def tanh(x):
    return math.tanh(x)


def deritanh(x):
    return 1 - math.tanh(x) ** 2


class neuron:
    def __init__(self, fun):
        self.valoue = 0
        self.newbias = 0
        self.fun = fun
        self.dh = 0
        self.raw_valoue = 0

    def roll(self, conections):
        self.weights = [round(1, 2) for x in range(conections)]
        self.dw = [0] * conections
        self.bias = round(np.random.normal(), 2)

    def calc(self, valous):
        val = sum([valous[x] * self.weights[x] for x in range(len(valous))]) + self.bias
        self.valoue = self.fun(val)
        self.raw_valoue = val
        return self.valoue


class two_neural:
    def __init__(self, learing=0.004, N=math.inf, layers=[2, 1], activation='sigmoid', beta=1, dropout_rate=0,
                 threshold=0.6):
        self.neurons = []
        self.set_fun(activation)
        self.layer = [0] + layers[0:-1]
        self.beta = beta
        self.dropout_rate = dropout_rate
        for i in layers:
            neuro = []
            for x in range(i):
                neuro.append(neuron(self.fun))
            self.neurons.append(neuro)
        self.learing = learing
        self.N = N
        self.set_fun(activation)
        self.plot = []
        self.acc = []
        self.threshold = threshold

    def set_fun(self, act):
        if act == 'sigmoid':
            self.fun = sigmoid
            self.derfun = derivsigmoid
        elif act == 'linear':
            self.fun = linear
            self.derfun = linear
        elif act == "relu":
            self.fun = relu
            self.derfun = relu
        elif act == 'tanh':
            self.fun = tanh
            self.derfun = deritanh

    def set_weights(self):
        for i in range(len(self.neurons)):
            for j in self.neurons[i]:
                j.roll(self.layer[i])

    def feed_forward(self, data_row, ydata):
        for i in self.neurons[0]:
            i.calc(data_row)

        for i in range(1, len(self.neurons)):
            vals = [self.neurons[i - 1][x].valoue for x in range(len(self.neurons[i - 1]))]
            for j in range(len(self.neurons[i])):
                self.neurons[i][j].calc(vals)
        return self.neurons[-1][0].valoue

    def calc_and_update(self, x, msedev):
        # calc new bias
        for i in self.neurons:
            for j in i:
                j.newbias = self.derfun(j.valoue)
        # calc dw
        for i in self.neurons[0]:
            for j in range(len(i.dw)):
                i.dw[j] = i.newbias * x[j]

        for i in range(1, len(self.neurons)):
            for j in range(len(self.neurons[i])):
                for k in range(len(self.neurons[i][j].dw)):
                    self.neurons[i][j].dw[k] = self.neurons[i][j].newbias * self.neurons[i - 1][k].valoue

        # calc dh
        for i in range(len(self.neurons) - 1):
            for j in range(len(self.neurons[i])):
                self.neurons[i][j].dh = 0
                for k in range(len(self.neurons[i + 1])):
                    self.neurons[i][j].dh += self.neurons[i + 1][k].newbias * self.neurons[i + 1][k].weights[j]

        # set new weights
        for i in range(len(self.neurons) - 1):
            for j in range(len(self.neurons[i])):
                for k in range(len(self.neurons[i][j].weights)):
                    if random.random() >= self.dropout_rate:
                        self.neurons[i][j].weights[k] -= self.learing * self.neurons[i][j].dw[k] * self.neurons[i][
                            j].dh * msedev

        for i in range(len(self.neurons[-1][0].weights)):
            if random.random() >= self.dropout_rate:
                self.neurons[-1][0].weights[i] -= self.learing * self.neurons[-1][0].dw[i] * msedev

        # set new bias
        for i in self.neurons:
            for j in i:
                if random.random() >= self.dropout_rate:
                    j.bias -= j.newbias * self.learing * msedev

    def train(self, xtrain, ytrain, epoch=10_000):
        self.acc = []
        self.plot = []
        self.layer[0] = len(xtrain[0])
        self.set_weights()
        for i in tqdm(range(1, epoch + 1)):
            error_sum = 0
            acc_sum = 0
            for x, y in zip(xtrain, ytrain):
                ypred = self.feed_forward(x, y)
                msedev = mse_derevative(y, ypred)
                error_sum += (ypred - y) ** 2
                ypred = 1 if ypred > self.threshold else 0
                acc_sum += 1 if ypred == y else 0
                self.calc_and_update(x, msedev[0])
            if i % self.N == 0:
                arr_ypred = []
                for x, y in zip(xtrain, ytrain):
                    arr_ypred.append(self.feed_forward(x, y))
                mse = mse_loss(ytrain, arr_ypred)

                print(ytrain, arr_ypred, mse)
            self.learing *= self.beta
            self.plot.append(error_sum / len(ytrain))
            self.acc.append(acc_sum / len(ytrain))

    def evaluate(self, xtest, ytest):
        answer = []
        for x, y in zip(xtest, ytest):
            ypred = self.feed_forward(x, y)
            if ypred > self.threshold:
                answer.append(1)
            else:
                answer.append(0)
            a = str(y)+"/"+str(ypred)
            # print(a)
        answertrue = [1 if answer[x] == ytest[x] else 0 for x in range(len(ytest))]
        return sum(answertrue) / len(answertrue)

    def show_plot_mse(self):
        df = pd.DataFrame({'epoch': list(range(1, len(self.plot) + 1)), "mse_loss": self.plot})
        fig = px.line(df, y="mse_loss", x="epoch")
        fig.show()

    def show_plot_acc(self):
        df = pd.DataFrame({'epoch': list(range(1, len(self.plot) + 1)), "acc": self.acc})
        fig = px.line(df, y="acc", x="epoch")
        fig.show()


if __name__ == "__main__":
    epoch = 10_000
    a = two_neural(learing=0.04, layers=[2, 2, 1])
    xtrain = [[1, 1], [0, 1], [1, 0], [0, 0]]
    ytrain = [1, 1, 1, 0]
    a.train(xtrain, ytrain, epoch)
    print("acc =", a.evaluate(xtrain, ytrain))
    a.show_plot_mse()
