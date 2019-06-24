#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# In[0]:
import numpy as np 
import math
import functools
import operator

import os
import glob
import cv2

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

class Layer:
    # modelにこのレイヤーが追加された時に呼ばれて, レイヤーを通した後のshapeを返す
    def layer_added(self, input_shape):
        return input_shape

class Linear(Layer):
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def layer_added(self, input_shape):
        self.input_shape = input_shape
        self.weights = np.random.rand(input_shape, self.output_shape)
        return self.output_shape

    def forward(self, x):
        return np.dot(x, self.weights)

    def backword(self, x, output_grad):
        input_grad = np.dot(output_grad, self.weights.T)
        change_grad = np.dot(x.T, output_grad)
        # TODO learning_rateを外部から与える
        learning_rate = 1e-5
        self.weights -= change_grad * learning_rate
        return input_grad

class ReLU(Layer):
    def forward(self, x):
        return np.maximum(x, 0)

    def derivative(self, x):
        x = np.where(x > 0, 1, 0)
        return x

    def backword(self, x, output_grad):
        return self.derivative(x) * output_grad

class Sigmoid(Layer):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return x * (1 - x)

    def backword(self, x, output_grad):
        return self.derivative(x) * output_grad

class NeuralNetwork:
    def __init__(self, input_shape, learning_rate=1e-5):

        prod = functools.partial(functools.reduce, operator.mul)
        self.output_shape = prod(input_shape)
        self.layers = []
        self.learning_rate = learning_rate

    def add(self, x):
        self.output_shape = x.layer_added(self.output_shape)
        self.layers.append(x)
        
    def feedforward(self):
        reshaped_input = self.x.reshape((self.x.shape[0], -1))

        self.layer_values = [reshaped_input]
        for layer in self.layers:
            self.layer_values.append(layer.forward(self.layer_values[-1]))

        return self.layer_values[-1]
        
    def backprop(self):
        reshaped_y = self.y.reshape((self.y.shape[0], -1))
        error = self.layer_values[-1] - reshaped_y
        loss = error * 2
        for i, layer in reversed(list(enumerate(self.layers))):
            loss = layer.backword(self.layer_values[i], loss)

    def train(self, x, y):
        self.x = x
        self.y = y
        self.feedforward()
        self.backprop()

    def test(self, x, output_shape=None):
        if output_shape is None:
            output_shape = (self.output_shape, )
        self.x = x
        output = self.feedforward()
        reshaped_output = output.reshape((-1, ) + output_shape)
        return reshaped_output
        
# In[1]:
x = np.array([[[0, 0], [1, 2]], [[2, 3], [3, 4]], [[4, 5], [1, 2]]], dtype=float)
y = np.array([[1], [2], [3]], dtype=float)

NN = NeuralNetwork((2, 2))
NN.add(Linear(200))
NN.add(ReLU())
NN.add(Linear(1))

epoch = 10000
for i in range(epoch):
    NN.train(x, y)
print("input " + str(x))
print ("Predicted Output: \n" + str(NN.test(x)))

'''

# In[2]:

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

# In[3]:
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

# In[4]:
pathes = glob.glob('./data/vegetables/*/*.jpg', recursive=True)
data_size = len(pathes)
classes = {}
class_cnt = 0
for path in pathes:
    class_name = os.path.dirname(path).split('/')[-1]
    if class_name not in classes:
        classes[class_name] = class_cnt
        class_cnt += 1

img_size = 24
inputs = np.zeros((data_size, img_size, img_size, 3))
outputs = np.zeros((data_size, class_cnt))
for idx, path in enumerate(pathes):
    class_name = os.path.dirname(path).split('/')[-1]
    img = cv2.imread(path)
    resized_img = cv2.resize(img, dsize=(img_size, img_size))
    inputs[idx] = resized_img
    outputs[idx][classes[class_name]] = 1

NN = NeuralNetwork(inputs, outputs)
epoch = 1000
for i in range(epoch):
    NN.train(inputs, outputs)

print('result : ', NN.test(inputs))
print('argmax : ', np.argmax(NN.test(inputs), axis=1))

#%%
'''