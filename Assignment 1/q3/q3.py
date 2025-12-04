#import libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 


#Note -> B is substitution for Thetha Parameters 


#import data 
x_train  = np.loadtxt("../data/Q3/logisticX.csv", delimiter=",")
y_train = np.loadtxt("../data/Q3/logisticY.csv", delimiter=",")
y_train = y_train.reshape((x_train.shape[0],1))


#data handling and normmalisation
X_train = np.ones((x_train.shape[0],x_train.shape[1]+1))
for i in range(x_train.shape[0]):
    for j in range(1,1+x_train.shape[1]):
        X_train[i][j] = x_train[i][j-1]

def normalise(X_train):
    mean_x_train = np.mean(X_train,axis = 0)
    std_x_train = np.std(X_train,axis = 0)
    mean_x_train[0] = 0
    std_x_train[0]  = 1
    for j in range(X_train.shape[1]):
        X_train[:,j] = (X_train[:,j] - mean_x_train[j])/std_x_train[j]
    return X_train
X_train = normalise(X_train)
B = np.zeros((1,X_train.shape[1]))     


#hession calculation 
def sigmoid(X,B):
    return 1/(1+np.exp(-X@B.T))

def log_liklehood(Y,X,B):
    L = 0
    for i in range(X.shape[0]):
        
        L += Y[i,0] * np.log(sigmoid(X[i],B)) + (1-Y[i,0]) * np.log(1-sigmoid(X[i],B))
    return L

def H_B(X,B):
    res = np.zeros((X.shape[0],1))
    for i in range(X.shape[0]):
        res[i,0] = sigmoid(X[i],B).item()
    return res    

def gradient_log_liklehood(Y,X,B):
    return (X.T @ (Y - H_B(X,B)))

def hessian(X,B):
    p = H_B(X,B).flatten()
    R = np.diag(p * (1 - p))
    return -(X.T @ (R @ X))

def hessian_inv(X,B):
    return np.linalg.inv(hessian(X,B))
        

#newton update 
def newtons_upadte(Y,X,e):
    B = np.zeros((1,X.shape[1]))
    grad_norm = np.linalg.norm(gradient_log_liklehood(Y,X,B))
    iter = 0
    while grad_norm>e:
        B = (B.T - (hessian_inv(X,B) @ gradient_log_liklehood(Y,X,B))).T
        grad_norm = np.linalg.norm(gradient_log_liklehood(Y,X,B))
        iter += 1
        print(f"iter = {iter} , B = {B}  , gradient_norm = {grad_norm} ")
    return B    

B_newton = newtons_upadte(y_train,X_train,1e-12)


#prediction 
p_predicted = H_B(X_train,B_newton) #probability calcualtion 
y_predicted = np.zeros(p_predicted.shape)

#implmenting boundary cnditions 
for i in range(p_predicted.shape[0]):
    if p_predicted[i] >= 0.5 :
        y_predicted[i] = 1



#plot data and boundary 
def draw_plots(y,X,B):
    y = y.flatten()
    plt.clf()
    plt.scatter(X[y==0, 1], X[y==0, 2], marker='o', color='red', label='Lable 0')
    plt.scatter(X[y==1, 1], X[y==1, 2], marker='x', color='blue', label='Lable 1')

    x1_vals = np.linspace(min(X[:,1])-1, max(X[:,1])+1, 100)
    x2_vals = -(B[0,0] + B[0,1]*x1_vals) / B[0,2]
    plt.plot(x1_vals, x2_vals, 'g-', label='Decision boundary')

    plt.xlabel('x1 normalised')
    plt.ylabel('x2 normalised')
    plt.legend()
    plt.title('Logistic Regression Decision Boundary')
    plt.show()

draw_plots(y_train,X_train,B_newton)

        