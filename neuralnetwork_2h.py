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
        self.learning_rate = 1e-1
        
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.x, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        self.layer3 = sigmoid(np.dot(self.layer2, self.weights3))
        return self.layer3
        
    def backprop(self):
        derivative_layer3 = sigmoid_derivative(self.layer3)
        derivative_layer2 = sigmoid_derivative(self.layer2)
        derivative_layer1 = sigmoid_derivative(self.layer1)

        error = self.layer3 - self.y 

        slope_layer3 = 2 * error * derivative_layer3
        d_weights3 = np.dot(self.layer2.T, slope_layer3)

        slope_layer2 = np.dot(slope_layer3, self.weights3.T) * derivative_layer2
        d_weights2 = np.dot(self.layer1.T, slope_layer2)

        slope_layer1 =np.dot(slope_layer2, self.weights2.T) * derivative_layer1
        d_weights1 = np.dot(self.x.T, slope_layer1)
    
        self.weights1 -= self.learning_rate * d_weights1
        self.weights2 -= self.learning_rate * d_weights2
        self.weights3 -= self.learning_rate * d_weights3

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

NN = NeuralNetwork(x,y)
epoch = 10000
for i in range(epoch): # trains the NN 1,000 times
    # if i % (epoch / 5) ==1: 
    #     print ("for iteration # " + str(i) + "\n")

    #     predict = NN.feedforward()
    #     print ("推論結果: \n" + str(predict))
    #     error = predict - y 
    #     print ("誤差 \n" + str(error))
    #     error_square = np.square(error)
    #     print ("誤差の二乗: \n" + str(error_square))
    #     #ここで言う損失関数は　np.mean(np.square(y - NN.feedforward()))
    #     print ("Loss: \n" + str(np.mean(error_square)))
    #     print ("\n")
  
    NN.train(x, y)

# In[2]:
x=np.array(([0, 0, 0.9],[0, 1.2, 1.1]), dtype=float)
print("input " + str(x))
print ("Predicted Output: \n" + str(NN.test(x)))
x=np.array(([1, 0, 1],[1, 1, 1]), dtype=float)
print("input " + str(x))
print ("Predicted Output: \n" + str(NN.test(x)))


x = 2 * math.pi * np.random.rand(100, 1)
y = np.sin(x)

NN = NeuralNetwork(x,y)
epoch = 10000
for i in range(epoch): 
    NN.train(x, y)


x=np.array(([[0],[math.pi/2],[math.pi],[3 * math.pi/2]]), dtype=float)
print("input " + str(x))
print ("Predicted Output: \n" + str(NN.test(x)))
