#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# In[0]:
import numpy as np 
import math

# 活性化関数（シグモイド関数）
def sigmoid(t):
    return 1/(1+np.exp(-t))

# シグモイド関数の微分
def sigmoid_derivative(p):
    return p * (1 - p)

# 活性化関数（relu関数）
def relu(t):
    return np.maximum(t, 0)

def relu_derivative(t):
    t = np.where(t > 0, 1, 0)
    return t

class NeuralNetwork:
    def __init__(self, x,y):
        self.weights1= np.random.rand(x.shape[1],4)
        self.weights2 = np.random.rand(4,4)
        self.weights3 = np.random.rand(4,y.shape[1])
        
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.x, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        self.layer3 = sigmoid(np.dot(self.layer2, self.weights3))
        return self.layer3
        
    def backprop(self):
        error = self.layer2 - self.y 
        derivative_layer3 = sigmoid_derivative(self.layer3)
        derivative_layer2 = sigmoid_derivative(self.layer2)
        derivative_layer1 = sigmoid_derivative(self.layer1)
        slope_layer3 = 2 * error * derivative_layer3
        d_weights3 = np.dot(self.layer2.T, slope_layer3)
        slope_layer2 = np.dot(slope_layer3, self.weights3.T) * derivative_layer2
        d_weights2 = np.dot(self.layer1.T, slope_layer2)
        slope_layer1 =np.dot(slope_layer2, self.weights2.T) * derivative_layer1
        d_weights1 = np.dot(self.x.T, slope_layer1)
    
        self.weights1 -= d_weights1
        self.weights2 -= d_weights2
        self.weights3 -= d_weights3

    def train(self, x, y):
        self.x = x
        self.y = y
        self.feedforward()
        self.backprop()

    def test(self, x):
        self.x = x
        output = self.feedforward()
        return output
        
# In[1]:

x=np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1]), dtype=float)
y=np.array(([0],[1],[1],[0]), dtype=float)
# x = 2 * math.pi * np.random.rand(100, 1)
# y = np.sin(x)

NN = NeuralNetwork(x,y)
for i in range(1000): # trains the NN 1,000 times
    if i % 200 ==1: 
        print ("for iteration # " + str(i) + "\n")

        predict = NN.feedforward()
        print ("推論結果: \n" + str(predict))
        error = predict - y 
        print ("誤差 \n" + str(error))
        error_square = np.square(error)
        print ("誤差の二乗: \n" + str(error_square))
        #ここで言う損失関数は　np.mean(np.square(y - NN.feedforward()))
        print ("Loss: \n" + str(np.mean(error_square)))
        print ("\n")
  
    NN.train(x, y)

# x=np.array(([[0],[math.pi/2],[math.pi],[3 * math.pi/2]]), dtype=float)
# print(x)
# print ("Predicted Output: \n" + str(NN.test(x)))

# In[2]:
x=np.array(([0, 0, 0.9],[0, 1.2, 1.1]), dtype=float)
y = NN.test(x)
print(y.shape)
print ("Predicted Output: \n" + str(NN.test(x)))
x=np.array(([1, 0, 1],[1, 1, 1]), dtype=float)
print ("Predicted Output: \n" + str(NN.test(x)))




#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# In[0]:
import numpy as np 
import math

# 活性化関数（例としてシグモイド関数）
def sigmoid(t):
    return 1/(1+np.exp(-t))

# シグモイド関数の微分
def sigmoid_derivative(p):
    return p * (1 - p)

# 活性化関数（例としてシグモイド関数）
def relu(t):
    return np.maximum(t, 0)

def relu_derivative(t):
    # print("t")
    # print(t)
    # t[False] = 0
    return t

class NeuralNetwork:
    def __init__(self, x,y):
        self.input = x
        self.weights1= np.random.rand(self.input.shape[1],4)
        self.weights2 = np.random.rand(4,4)
        self.weights3 = np.random.rand(4,1)
        self.y = y
        self.output = np.zeros(y.shape)
        
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        self.layer3 = sigmoid(np.dot(self.layer2, self.weights3))
        # layer2がアウトプットになる
        return self.layer2
        
    def backprop(self):
        error = self.y -self.output
        slope_layer3 = sigmoid_derivative(self.output)
        slope_layer2 = sigmoid_derivative(self.layer2)
        slope_layer1 = sigmoid_derivative(self.layer1)
        d_weights3 = np.dot(self.layer1.T, 2*(error)*slope_layer3)
        d_weights2 = np.dot(self.input.T, np.dot(2*(error)*slope_layer3, self.weights2.T)*slope_layer2)
        d_weights1 = np.dot(self.input.T, np.dot(2*(error)*slope_layer2, self.weights2.T)*slope_layer1)
    
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        self.weights3 += d_weights3

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()

    def test(self, x):
        self.input = x
        output = self.feedforward()
        return output
        
# In[1]:

x=np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1]), dtype=float)
y=np.array(([0],[1],[1],[0]), dtype=float)

x = 2 * math.pi * np.random.rand(100, 1)
y = np.sin(x)

NN = NeuralNetwork(x,y)
for i in range(1000): # trains the NN 1,000 times
    if i % 200 ==0: 
        print ("for iteration # " + str(i) + "\n")
        predict = NN.feedforward()
        print ("推論結果: \n" + str(predict))
        error = y - predict
        print ("誤差 \n" + str(error))
        error_square = np.square(error)
        print ("誤差の二乗: \n" + str(error_square))
        #ここで言う損失関数は　np.mean(np.square(y - NN.feedforward()))
        print ("Loss: \n" + str(np.mean(error_square)))
        print ("\n")
  
    NN.train(x, y)

x=np.array(([[0],[math.pi/2],[math.pi],[3 * math.pi/2]]), dtype=float)
print(x)
print ("Predicted Output: \n" + str(NN.test(x)))

# In[2]:
# x=np.array(([0,0,0.9],[0,1.2,1.1]), dtype=float)
# print ("Predicted Output: \n" + str(NN.test(x)))
# X=np.array(([1,0,1],[1,1,1]), dtype=float)
# print ("Predicted Output: \n" + str(NN.test(x)))
