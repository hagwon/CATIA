#import tensorflow as tf
import math
import random

input_layer = [0.0, 0.0]
hidden_layer = [0.0, 0.0, 0.0, 0.0]
output_layer = [0.0, 0.0]

weightZ = [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
weightY = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]

for i in range(4):
    for j in range(2):
        weightZ[j][i] = round(random.uniform(-10000, 10000) / 10000, 5)
        weightY[i][j] = round(random.uniform(-10000, 10000) / 10000, 5)

print(weightZ)
print(weightY)

n = 0.2 # 학습률

def SquaredError(d, y):
    temp = math.pow(d-y, 2)

    return temp / 2

def Optimizer(w, ac, aw, k):
    temp = w - n*(ac/aw)
    k = k + 1

    return temp, k

def Sigmoid(u):
    temp = 1.0 / (1.0 + math.exp(-1 * u))

    if temp > 1:
        temp = 1
    elif temp < 0:
        temp = 0

    return temp

def ErrorSignalY(d, y):
    return (d - y) * y * (1 - y)

def ErrorSignalZ(dy, w, z):
    sig = 0
    for i in (0, 2):
        sig = sig + (dy * w)
    temp = z * (1-z) * sig

    return temp

def InputLayer(f, s):
    input_layer[0] = f
    input_layer[1] = s

def CalculateWeight(inputIndex, weightIndex1, weightIndex2):
    return input_layer[inputIndex] * weightZ[weightIndex1][weightIndex2]

def CalculateWeight2(hiddenIndex, weightIndex1, weightIndex2):
    return hidden_layer[hiddenIndex] * weightY[weightIndex1][weightIndex2]

def HiddenLayer(w1, w2):
    temp = 0
    for i in range(2):
        temp = w1 + w2

    return temp

def OutputLayer(w1, w2, w3, w4):
    
    temp = 0
    for i in range(4):
        temp = w1 + w2 + w3 + w4

    return temp

InputLayer(5, 5)
for i in range(4):
    for j in range(2):
        w1 = CalculateWeight(0, j, i)
        w2 = CalculateWeight(1, j, i)
        h1 = HiddenLayer(w1, w2)
        hidden_layer[i] = Sigmoid(h1)

for i in range(2):
    for j in range(4):
        w1 = CalculateWeight2(0, j, i)
        w2 = CalculateWeight2(1, j, i)
        w3 = CalculateWeight2(2, j, i)
        w4 = CalculateWeight2(3, j, i)
        o1 = OutputLayer(w1, w2, w3, w4)
        output_layer[i] = Sigmoid(o1)

print(output_layer)
