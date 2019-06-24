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

class NeuralNetwork:
    def __init__(self, x, y):
        self.input_shape = x[0].shape
        self.output_shape = y[0].shape
        self.x = x
        self.y = y
        prod = functools.partial(functools.reduce, operator.mul)
        self.weights1= np.random.rand(prod(self.input_shape), 4)
        self.weights2 = np.random.rand(4, prod(self.output_shape))
        self.learning_rate = 1e-5
        
    def feedforward(self):
        reshaped_x = self.x.reshape((self.x.shape[0], -1))
        self.layer1 = relu(np.dot(reshaped_x, self.weights1))
        self.layer2 = relu(np.dot(self.layer1, self.weights2))
        # layer2がアウトプットになる
        return self.layer2
        
    def backprop(self):
        reshaped_x = self.x.reshape((self.x.shape[0], -1))
        reshaped_y = self.y.reshape((self.y.shape[0], -1))
        error = self.layer2 - reshaped_y
        derivative_layer2 = relu_derivative(self.layer2)
        derivative_layer1 = relu_derivative(self.layer1)

        slope_layer2 = 2 * error * derivative_layer2
        d_weights2 = np.dot(self.layer1.T, slope_layer2)

        slope_layer1 = np.dot(slope_layer2, self.weights2.T) * derivative_layer1
        d_weights1 = np.dot(reshaped_x.T, slope_layer1)
    
        self.weights1 -= self.learning_rate * d_weights1
        self.weights2 -= self.learning_rate * d_weights2

    def train(self, x, y):
        self.x = x
        self.y = y
        self.feedforward()
        self.backprop()

    def test(self, x):
        self.x = x
        output = self.feedforward()
        reshaped_output = output.reshape((-1, ) + self.output_shape)
        return reshaped_output
        
# In[1]:
x = np.array([[[0, 0], [1, 2]], [[2, 3], [3, 4]], [[4, 5], [1, 2]]], dtype=float)
y = np.array([[1], [2], [3]], dtype=float)

NN = NeuralNetwork(x, y)
epoch = 10000
for i in range(epoch):
    NN.train(x, y)
print("input " + str(x))
print ("Predicted Output: \n" + str(NN.test(x)))
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
