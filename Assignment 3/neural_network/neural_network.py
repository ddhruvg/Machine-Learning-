import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
class Layer:
    def __init__(self,in_dimension , out_dimension , learning_rate = 0.001 , is_sigmoid = True,is_output = False):
        self.net_grad = None
        self.lr = learning_rate
        self.O = None
        limit = np.sqrt(6 / (in_dimension + out_dimension))
        self.w = np.random.uniform(-limit, limit, (in_dimension + 1, out_dimension)).astype(np.float32)

        self.output_function = self.__sigmoid if is_sigmoid else self.__relu
        self.output_function_grad = self.__sigmoid_grad if is_sigmoid else self.__relu_grad
        self.is_output = is_output

    def __relu(self,x):
        return np.maximum(0,x)

    def __relu_grad(self,x):
        return np.where(x > 0 , 1 , 0)

    def __sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def __softmax(self,x):
        exp_x = np.exp(x - np.max(x , axis =1 , keepdims=True))
        return exp_x / np.sum(exp_x, axis = 1 , keepdims=True)

    def __sigmoid_grad(self,x):
        return x * (1-x)    

    def next(self,X ):
        X = np.hstack([np.ones((X.shape[0] , 1)),X])
        self.net_grad = np.dot(X, self.w)
        self.O = self.output_function(self.net_grad)
        return self.O

    def _prev(self, dO , O_down):
        if not self.is_output:
            new_grad = dO * self.output_function_grad(self.O)
            dw = np.dot(np.hstack([np.ones((O_down.shape[0], 1)), O_down]).T , new_grad) / O_down.shape[0]
            d_O_down = np.dot(new_grad , self.w.T)
            dO_prev =   d_O_down[:,1:]
            self.w = self.w - self.lr * dw
            return  dO_prev
        
            
    

class FinalLayer :
    def __init__(self,in_dim , out_dim , lr = 0.001 ,is_softmax = True) :
        self.lr = lr
        limit = np.sqrt(6 / (in_dim + out_dim))
        self.w = np.random.uniform(-limit, limit, (in_dim + 1, out_dim)).astype(np.float32)
        self.net_grad = None
        self.O = None
        self.output_function = self.__softmax if is_softmax else self.__sigmoid

    
           

    def __sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def __softmax(self,x):
        exp_x = np.exp(x - np.max(x , axis =1 , keepdims=True))
        return exp_x / np.sum(exp_x, axis = 1 , keepdims=True)

    def __sigmoid_grad(self,x):
        return x * (1-x)  

    def next(self , X) :
        X = np.hstack([np.ones((X.shape[0] , 1)),X ])    
        self.net_grad = np.dot(X, self.w)
        self.O = self.output_function(self.net_grad)
        return self.O
    
    def _prev(self, y_true , O_down)  :
        new_grad = self.O - y_true
        dw = np.dot(np.hstack([np.ones((O_down.shape[0], 1)), O_down]).T , new_grad) / y_true.shape[0]
        d_O_down = np.dot(new_grad , self.w.T)
        dO_prev =   d_O_down[:,1:]
        self.w = self.w - self.lr * dw
        return dO_prev
    
class NeuralNetwork:
    def __init__(self, M = 32 , n = 3072 , HiddenLayer = [512 , 265 ]  , target_class = 36 , lr = 0.01,is_sigmoid = True):
        self.lr = lr
        self.layers = []
        self.Batch_size = M
        prev_dimn = n
        for h in HiddenLayer:
            layer = Layer(prev_dimn , h , is_sigmoid = is_sigmoid , learning_rate = self.lr)
            self.layers.append(layer)
            prev_dimn = h
        self.output_layer = FinalLayer(prev_dimn , target_class , is_softmax = True , lr = self.lr)

    def next(self,X) : 
        O_down = X 
        for layer in self.layers :
            O_down = layer.next(O_down)

        predictions = self.output_layer.next(O_down)
        return predictions
    
    def corss_entropy_loss(self, y_true , y_pred):
        m = y_true.shape[0]
        return - np.sum(y_true * np.log(y_pred + 1e-10 )) / m
     
    def _prev(self,Y , X) : 
        O = X 
        stored_outputs = [X]
        for layer in self.layers :
            O = layer.next(O)
            stored_outputs.append(O)
        predictions = self.output_layer.next(O)

       

        do_down = self.output_layer._prev(Y , stored_outputs[-1])
        for i in range(len(self.layers)-1 , -1 , -1):
            do_down = self.layers[i]._prev(do_down , stored_outputs[i])

        return predictions  

    def predict(self,X) :
        predictions = self.next(X)
        return np.argmax(predictions , axis = 1)   
    
    def fit(self , X , Y , epochs = 10 ):
        for epoch in range(epochs) : 
            comb = np.random.permutation(X.shape[0])
            X_shuffled = X[comb]
            Y_shuffled = Y[comb]
            J = 0

            for i in range(0, X.shape[0] , self.Batch_size):
                X_batch = X_shuffled[i : i + self.Batch_size]
                Y_batch = Y_shuffled[i : i + self.Batch_size]
                predictions = self._prev(Y_batch , X_batch)
                loss = self.corss_entropy_loss(Y_batch , predictions)
                J += loss
           
            # print(f"Epoch {epoch + 1} / {epochs} , Loss : {J}")
