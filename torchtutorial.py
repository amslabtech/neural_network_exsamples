# -*- coding: utf-8 -*-
# In[0]:
import numpy as np
import math

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
D_in, H, D_out = 3, 100, 1
D_in, H, D_out = 1, 100, 1


class NeuralNetwork:
    def __init__(self, x,y):
        self.x = x
        # Randomly initialize weights
        self.w1 = np.random.randn(D_in, H)
        self.w2 = np.random.randn(H, D_out)
        self.y = y
        self.output = np.zeros(y.shape)
        # self.learning_rate = 1e-5
        self.learning_rate = 0.1
        
    def feedforward(self):
        # Forward pass: compute predicted y
        self.h = x.dot(self.w1)
        self.h_relu = np.maximum(self.h, 0)
        self.y_pred = self.h_relu.dot(self.w2)
        return self.y_pred
        
    def backprop(self):
        
        grad_y_pred = 2.0 * (self.y_pred - self.y)
        grad_w2 = self.h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(self.w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[self.h < 0] = 0
        grad_w1 = x.T.dot(grad_h)

        # Update weights
        self.w1 -= self.learning_rate * grad_w1
        self.w2 -= self.learning_rate * grad_w2

    def train(self, x, y):
        self.y_pred = self.feedforward()
        self.backprop()

    def test(self, x):
        self.x = x
        y_pred = self.feedforward()
        return y_pred

# In[1]:
# Create random input and output data
# x=np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1]), dtype=float)
# y=np.array(([0],[1],[1],[0]), dtype=float)

# x =  np.random.randint(0, 100, (N, D_in)) 
# y = np.random.randn(N, D_out)
x = 2 * math.pi * np.random.rand(100, 1)
y = np.sin(x)

print(x)
print(y)
# In[2]:
NN = NeuralNetwork(x,y)

for i in range(3000): # trains the NN 1,000 times
    if i % 200 ==0: 
        predict = NN.feedforward()
        error = y - predict
        error_square = np.square(error)
        # print ("for iteration # " + str(i) + "\n")
        # print ("推論結果: \n" + str(predict))
        # print ("誤差 \n" + str(error))
        # print ("誤差の二乗: \n" + str(error_square))
        #ここで言う損失関数は　np.mean(np.square(y - NN.feedforward()))
        print ("Loss: \n" + str(np.mean(error_square)))
        print ("\n")
  
    NN.train(x, y)

# In[3]:
# X=np.array(([0,0,0.9],[0,1.2,1.1]), dtype=float)
# print ("Predicted Output: \n" + str(NN.test(X)))
# X=np.array(([1,0,1],[1,1,1]), dtype=float)
# print ("Predicted Output: \n" + str(NN.test(X)))


x=np.array(([[0],[math.pi/2],[3 * math.pi/2]]), dtype=float)
print(x)
print ("Predicted Output: \n" + str(NN.test(x)))

#%%
