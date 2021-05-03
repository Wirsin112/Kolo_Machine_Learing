import pandas as pd
import random
import math
import statistics as st
#Jakbym cos zjebal

def sigmoid(x):
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
    return 1 - math.tanh(x)**2
class neuron:
    def __init__(self, fun):
        self.valoue = 0
        self.weights = [round(random.random(), 2) for x in range(2)]
        self.bias = round(random.random(), 2)
        self.fun = fun
    def calc(self, valous):
        val = sum([valous[x] * self.weights[x] for x in range(len(valous))]) + self.bias
        self.valoue = self.fun(val)
        return self.valoue

"""
Funkcje aktywacji sigmoid, linera, relu, tanh,
Neurony zrobić tak żeby działały i będzie pog
"""
class two_neural:
    def __init__(self, learing=0.0004, N = math.inf, layers = [2,1] , activation= 'sigmoid'):
        self.neurons = []
        self.set_fun(activation)
        for i in layers:
            neuro = []
            for x in range(i):
                neuro.append(neuron(self.fun))
            self.neurons.append(neuro)
        print(self.neurons)
        self.learing = learing
        self.N = N

    def set_fun(self, act):
        if act == 'sigmoid':
            self.fun = sigmoid
            self.derfun = derivsigmoid
        elif act == 'linear':
            self.fun = linear
            self.derfun = linear
        elif act == 'relu':
            self.fun = relu
            self.derfun = relu
        elif act == 'tanh':
            self.fun = tanh
            self.derfun = deritanh

    def feed_forward(self, data_row, ydata):
        vals = []
        for x in self.neurons[0]:
            vals.append(x.calc(data_row))
        return self.neurons[1][0].calc(vals)

    def calc_and_update(self, x, msedev):
        dw5 = self.neurons[0][1].valoue * self.derfun(self.neurons[1][0].valoue)
        dw4 = self.neurons[0][0].valoue * self.derfun(self.neurons[1][0].valoue)

        dh0 = self.neurons[1][0].weights[0] * self.derfun(self.neurons[1][0].valoue)
        dh1 = self.neurons[1][0].weights[1] * self.derfun(self.neurons[1][0].valoue)
        dw0 = x[0] * self.derfun(self.neurons[0][0].valoue)
        dw1 = x[1] * self.derfun(self.neurons[0][0].valoue)

        dw2 = x[0] * self.derfun(self.neurons[0][1].valoue)
        dw3 = x[1] * self.derfun(self.neurons[0][1].valoue)

        b2 = self.derfun(self.neurons[1][0].valoue)
        b0 = self.derfun(self.neurons[0][0].valoue)
        b1 = self.derfun(self.neurons[0][1].valoue)

        # update
        self.neurons[0][0].weights[0] = self.neurons[0][0].weights[0] - self.learing * dw0 * dh0 * msedev
        self.neurons[0][0].weights[1] = self.neurons[0][0].weights[1] - self.learing * dw1 * dh0 * msedev

        self.neurons[0][1].weights[0] = self.neurons[0][1].weights[0] - self.learing * dw2 * dh1 * msedev
        self.neurons[0][1].weights[1] = self.neurons[0][1].weights[1] - self.learing * dw3 * dh1 * msedev

        self.neurons[1][0].weights[0] = self.neurons[1][0].weights[0] - self.learing * dw4 * msedev
        self.neurons[1][0].weights[1] = self.neurons[1][0].weights[1] - self.learing * dw5 * msedev

        self.neurons[0][0].bias = self.neurons[0][0].bias - b0 * self.learing * msedev
        self.neurons[0][1].bias = self.neurons[0][1].bias - b1 * self.learing * msedev
        self.neurons[1][0].bias = self.neurons[1][0].bias - b2 * self.learing * msedev

    def train(self, xtrain, ytrain, epochs):
        for i in range(1, epoch + 1):
            for x, y in zip(xtrain, ytrain):
                ypred = self.feed_forward(x, y)
                msedev = mse_derevative(y, ypred)
                self.calc_and_update(x, msedev[0])
            if i % self.N == 0:
                arr_ypred = []
                for x, y in zip(xtrain, ytrain):
                    arr_ypred.append(self.feed_forward(x, y))
                mse = mse_loss(ytrain, arr_ypred)
                print(ytrain, arr_ypred, mse)
    def evaluate(self, xtest, ytest):
        answer = []
        for x,y in zip(xtest, ytest):
            ypred = self.feed_forward(x, y)
            if ypred>0.8:
                answer.append(1)
            else:
                answer.append(0)
            print(x,"=",y,"/",ypred)
        answertrue = [1 if answer[x] == ytest[x] else 0 for x in range(len(ytest))]
        return sum(answertrue)/len(answertrue)

if __name__ == "__main__":
    epoch = 50_000
    a = two_neural(learing=0.004)
    xtrain = [[1, 1], [1, 0], [0, 1], [0, 0]]
    ytrain = [1, 1, 1, 0]
    a.train(xtrain, ytrain, epoch)
    print(a.evaluate([[0,1],[0,1],[1,0],[0,0],[1,1],[1,1],[0,0],[0,1],[1,1],[1,0]],[1,1,1,0,1,1,0,1,1,1]))