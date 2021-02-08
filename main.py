import math
import numpy as np


def sigmoid(x):
    am = 1.0 / (1 + np.exp(-x))
    return am


class multilayer_perceptron:
    def __init__(self, inp, hid, out):
        # network initial
        self.input_size = inp
        self.output_size = out
        self.hidden_size = hid
        self.hidden = {}
        self.weights = {}

        self.desired = np.zeros(self.output_size)  # desired output
        self.input = np.zeros(self.input_size)  # network input
        self.output = np.zeros(self.output_size)  # network output
        for i, hiddenSize in enumerate(self.hidden_size):  # network hidden layer
            self.hidden[i] = np.zeros(hiddenSize)
        self.deltaOut = (np.zeros(self.output_size))  # delta (change) of outer weight

        # weight initial
        first_size = self.input_size
        for i, hiddenSize in enumerate(self.hidden_size):
            second_size = hiddenSize
            self.weights[i] = np.random.uniform(-1, 1, (first_size, second_size)) * math.sqrt(
                6. / (first_size + second_size))
            first_size = second_size
        self.weights[len(self.hidden_size)] = np.random.uniform(-1, 1, (first_size, self.output_size)) * math.sqrt(
            6. / (first_size + self.output_size))

    def set_input(self, x):
        self.input = x[0:self.input_size]
        self.desired = np.zeros(self.output_size)

        for i in range(0, self.output_size):
            if i == x[self.input_size]:
                self.desired[i] = 1
            else:
                self.desired[i] = 0

    def feed_forward(self):
        previous_layer = self.input
        for i in range(len(self.hidden)):
            for j in range(len(self.hidden[i])):
                self.hidden[i] = sigmoid(np.dot(previous_layer, self.weights[i]))
            previous_layer = self.hidden[i]
        self.output = sigmoid(np.dot(previous_layer, self.weights[len(self.hidden)]))

    def backpropagation(self):
        total_error_rate = np.mean(np.square(self.desired - self.output))

        # outer layer
        deltaW = np.zeros((self.hidden_size[len(self.hidden_size) - 1], self.output_size))
        deltaOut = (np.zeros(self.output_size))
        for i in range(self.output_size):
            deltaOut[i] = (self.output[i] - self.desired[i]) * (self.output[i]) * (1 - self.output[i])
            for j in range(self.hidden_size[len(self.hidden_size) - 1]):
                deltaW[j, i] = deltaOut[i] * self.hidden[len(self.hidden_size) - 1][j]
        self.weights[len(self.hidden_size)] = self.weights[len(self.hidden_size)] - 0.5 * deltaW

        # hidden layer
        for z in range(len(self.hidden_size) - 1):
            deltaW = np.zeros(
                (self.hidden_size[len(self.hidden_size) - 2 - z], self.hidden_size[len(self.hidden_size) - 1 - z]))
            deltaOut2 = (np.zeros(self.hidden_size[len(self.hidden_size) - 1 - z]))
            for i in range(self.hidden_size[len(self.hidden_size) - 1 - z]):
                deltaOut2[i] = (self.hidden[len(self.hidden_size) - 1 - z][i]) * (
                        1 - self.hidden[len(self.hidden_size) - 1 - z][i]) * np.dot(
                    self.weights[len(self.hidden_size) - z][i], deltaOut)
                for j in range(self.hidden_size[len(self.hidden_size) - 2 - z]):
                    deltaW[j, i] = deltaOut2[i] * self.hidden[len(self.hidden_size) - 2 - z][j]
            self.weights[len(self.hidden_size) - 1 - z] = self.weights[len(self.hidden_size) - 1 - z] - 0.5 * deltaW
            deltaOut = deltaOut2

        # input layer
        deltaW = np.zeros((self.input_size, self.hidden_size[0]))
        deltaOut2 = (np.zeros(self.hidden_size[0]))
        for i in range(self.hidden_size[0]):
            deltaOut2[i] = (self.hidden[0][i]) * (1 - self.hidden[0][i]) * np.dot(self.weights[1][i], deltaOut)
            for j in range(self.input_size):
                deltaW[j, i] = deltaOut2[i] * self.input[j]
        self.weights[0] = self.weights[0] - 0.5 * deltaW

    def test(self, x):
        self.input = x[0:self.input_size]
        for i in range(0, self.output_size):
            if i == x[self.input_size]:
                self.desired[i] = 1
            else:
                self.desired[i] = 0

        previous_layer = self.input
        for i in range(len(self.hidden)):
            for j in range(len(self.hidden[i])):
                self.hidden[i] = sigmoid(np.dot(previous_layer, self.weights[i]))
            previous_layer = self.hidden[i]
        self.output = sigmoid(np.dot(previous_layer, self.weights[len(self.hidden)]))
        x = -100000
        y = 0
        for i, k in enumerate(self.output):
            if k > x:
                y = i
                x = k

        return y


def loadFile():
    out = []
    file = open('./train.txt', encoding="utf8")

    for line in file:
        line = line.replace(' ', '')
        records = line.split(',')
        records.pop()
        newRec = []
        for record in records:
            newRec.append(float(record))
        out.append(newRec)
    file.close()
    return out


def loadFile2():
    out = []
    file = open('./test.txt', encoding="utf8")

    for line in file:
        line = line.replace(' ', '')
        records = line.split(',')
        records.pop()
        newRec = []
        for record in records:
            newRec.append(float(record))
        out.append(newRec)
    file.close()
    return out


data = np.asarray(loadFile())
test = np.asarray(loadFile2())

maxARR = np.max(a=data, axis=0)
minARR = np.min(a=data, axis=0)
data = (data - minARR) / (maxARR - minARR)
test = (test - minARR) / (maxARR - minARR)

import matplotlib.pyplot as plt

np.random.shuffle(data)
train_index = data[:len(data) // 5]
test_index = data[len(data) // 5:]
N = multilayer_perceptron(14, [9, 7], 2)

validationSet = []
testSet = []
ep = []
for i in range(0, 100):
    print("Epoch:", str(i))
    x = 0
    for record in test_index:
        if record[14] == N.test(record):
            x += 1
    print((x / len(test_index)) * 100, "%")
    validationSet.append((x / len(test_index)) * 100)
    for record in train_index:
        N.set_input(record)
        N.feed_forward()
        N.backpropagation()

    x = 0
    for record in test:
        if record[14] == N.test(record):
            x += 1
    print("FINAL:", (x / len(test)) * 100, "%")
    testSet.append((x / len(test)) * 100)
    ep.append(i)
plt.plot(ep, validationSet, 'r--', ep, testSet, 'b--')
plt.savefig('books_read.png')
