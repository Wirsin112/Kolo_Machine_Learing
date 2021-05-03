import pandas as pd
import random
import math
import statistics as st
from tqdm import tqdm

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def derivsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mse_loss(y, ypred):
    return st.mean([(y[x] - ypred[x]) ** 2 for x in range(len(y))])


def derivmse(y, ypred):
    return -2 * (y - ypred)


def mse_derevative(y, ypred):
    return -2 * (y - ypred),


class neuron:
    def __init__(self):
        self.valoue  = 0
        self.weights = [round(random.random(), 2) for x in range(2)]
        self.bias = round(random.random(), 2)
        self.raw_val = 0
    def calc(self, valous):
        val = sum([valous[x] * self.weights[x] for x in range(len(valous))]) + self.bias
        self.valoue = sigmoid(val)
        self.raw_val = val
        return self.valoue


class two_neural:
    def __init__(self, learing=0.0004, N=math.inf):
        self.neurons = [[neuron(), neuron()], neuron()]
        self.learing = learing
        self.N = N

    def feed_forward(self, data_row, ydata):
        vals = []
        for x in self.neurons[0]:
            vals.append(x.calc(data_row))
        return self.neurons[1].calc(vals)

    def calc_and_updtae(self, x, msedev):
        dw5 = self.neurons[0][1].valoue * derivsigmoid(self.neurons[1].raw_val)
        dw4 = self.neurons[0][0].valoue * derivsigmoid(self.neurons[1].raw_val)

        dh0 = self.neurons[1].weights[0] * derivsigmoid(self.neurons[1].raw_val)
        dh1 = self.neurons[1].weights[1] * derivsigmoid(self.neurons[1].raw_val)
        dw0 = x[0] * derivsigmoid(self.neurons[0][0].raw_val)
        dw1 = x[1] * derivsigmoid(self.neurons[0][0].raw_val)

        dw2 = x[0] * derivsigmoid(self.neurons[0][1].raw_val)
        dw3 = x[1] * derivsigmoid(self.neurons[0][1].raw_val)

        b2 = derivsigmoid(self.neurons[1].raw_val)
        b0 = derivsigmoid(self.neurons[0][0].raw_val)
        b1 = derivsigmoid(self.neurons[0][1].raw_val)

        # update
        self.neurons[0][0].weights[0] = self.neurons[0][0].weights[0] - self.learing * dw0 * dh0 * msedev
        self.neurons[0][0].weights[1] = self.neurons[0][0].weights[1] - self.learing * dw1 * dh0 * msedev

        self.neurons[0][1].weights[0] = self.neurons[0][1].weights[0] - self.learing * dw2 * dh1 * msedev
        self.neurons[0][1].weights[1] = self.neurons[0][1].weights[1] - self.learing * dw3 * dh1 * msedev

        self.neurons[1].weights[0] = self.neurons[1].weights[0] - self.learing * dw4 * msedev
        self.neurons[1].weights[1] = self.neurons[1].weights[1] - self.learing * dw5 * msedev

        self.neurons[0][0].bias = self.neurons[0][0].bias - b0 * self.learing * msedev
        self.neurons[0][1].bias = self.neurons[0][1].bias - b1 * self.learing * msedev
        self.neurons[1].bias = self.neurons[1].bias - b2 * self.learing * msedev

    def train(self, xtrain, ytrain, epochs):
        for i in tqdm(range(1, epoch + 1)):
            for x, y in zip(xtrain, ytrain):
                ypred = self.feed_forward(x, y)
                msedev = mse_derevative(y, ypred)
                self.calc_and_updtae(x, msedev[0])
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
    epoch = 800000
    a = two_neural(learing=0.04)
    xtrain = [[1, 1], [1, 0], [0, 1], [0, 0]]
    ytrain = [1, 0, 0, 1]
    a.train(xtrain, ytrain, epoch)
    print(a.evaluate(xtrain,ytrain))