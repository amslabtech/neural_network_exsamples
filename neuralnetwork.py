#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# In[0]:
import numpy as np 

# 活性化関数（例としてシグモイド関数）
def sigmoid(t):
    return 1/(1+np.exp(-t))

# シグモイド関数の微分
def sigmoid_derivative(p):
    return p * (1 - p)

class NeuralNetwork:
    def __init__(self, x,y):
        self.input = x
        self.weights1= np.random.rand(self.input.shape[1],4)
        self.weights2 = np.random.rand(4,1)
        self.y = y
        self.output = np.zeros(y.shape)
        
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        # layer2がアウトプットになる
        return self.layer2
        
    def backprop(self):
        error = self.y -self.output
        slope_layer2 = sigmoid_derivative(self.output)
        slope_layer1 = sigmoid_derivative(self.layer1)
        d_weights2 = np.dot(self.layer1.T, 2*(error)*slope_layer2)
        d_weights1 = np.dot(self.input.T, np.dot(2*(error)*slope_layer2, self.weights2.T)*slope_layer1)
    
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()

    def test(self, x):
        self.input = x
        output = self.feedforward()
        return output
        
# In[1]:

X=np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1]), dtype=float)
y=np.array(([0],[1],[1],[0]), dtype=float)

NN = NeuralNetwork(X,y)
for i in range(1000): # trains the NN 1,000 times
    if i % 200 ==0: 
        print ("for iteration # " + str(i) + "\n")
        predict = NN.feedforward()
        print ("推論結果: \n" + str(predict))
        error = y - predict
        print ("誤差 \n" + str(error))
        error_square = np.square(error)
        print ("誤差の二乗: \n" + str(error_square))
        print ("Loss: \n" + str(np.mean(error_square)))
        print ("\n")
  
    NN.train(X, y)


# In[2]:
X=np.array(([0,0,0.9],[0,1.2,1.1]), dtype=float)
print ("Predicted Output: \n" + str(NN.test(X)))
X=np.array(([1,0,1],[1,1,1]), dtype=float)
print ("Predicted Output: \n" + str(NN.test(X)))
