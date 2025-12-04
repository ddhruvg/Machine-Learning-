#load librraies 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from mpl_toolkits.mplot3d import Axes3D 

#Note -> B is substitution for Thetha Parameters 

#Data manufacturing and handling 
total_sample = 1000000
x1 = np.random.normal(3,2,total_sample)
x2 = np.random.normal(-1,2,total_sample)
e = np.random.normal(0,2,total_sample)
B = np.array([[3,1,2]])
X = np.ones((total_sample,3))
Y = np.ones((total_sample,1))

for i in range(1,3):
    for j in range(total_sample):
        if i == 1:
            X[j,i] = x1[j]
        else:
            X[j,i] = x2[j]

for j in range(total_sample):
    Y[j,0] = (X[j]@B.T).item() + e[j]

#Train test split 
Y_train = Y[:int(0.8 * total_sample),:]
X_train = X[:int(0.8 * total_sample),:]
Y_test = Y[int(0.8 * total_sample)+1:,:]
X_test = X[int(0.8 * total_sample)+1:,:]    



#sgd fucntions 
def hypothesis(X,B):
    return X@B.T

def loss_fun(X,Y,B):
    m = len(Y)
    H = hypothesis(X,B)
    error = (Y - H)
    return (error.T@ error )/(2*m)


def stoachstic_gradient_descent(r,alpha,epoch,X,Y):
    idx = np.random.permutation(X_train.shape[0])
    X = X[idx]
    Y = Y[idx]
    B = np.zeros((1,X.shape[1]))
    

    m = len(X)
    delta = float('inf')
    cnt  = 0

    loss =  loss_fun(X[0:r ,: ],Y[0:r ,: ],B)

 
    
    for ep in range(epoch):
       
        for cnt in range(m//r):

            start = (cnt * r) 
            end = min(start + r, m) 
            X_epoch = X[start:end, :]
            Y_epoch = Y[start:end, :]
            H_epoch = hypothesis(X_epoch,B)

            
            grad = np.dot((Y_epoch - H_epoch).T, X_epoch) / len(X_epoch)
            nrm = np.linalg.norm(alpha*grad)
            
            if nrm < 1e-6 or int((ep ) * m//r + cnt) > 1e5  : # make nrm < 1e-10 for r == 1 beacuse 1e-6 might not give correct convergence for r = 1
             
                print(f"total iter = {(ep ) * m//r + cnt} , B = {B} ,norm = {nrm} ")
                return B
            B = B + alpha * grad

            delta = abs(loss - loss_fun(X_epoch,Y_epoch,B)) #difference in updated value of loss 
            loss = loss_fun(X_epoch,Y_epoch,B)
            
            cnt += 1 
           
            

        # print(f"epoch = {ep}  , B = {B[0]} " , f"Loss = {loss}" ,f"Delta = {delta} , norm = {np.linalg.norm(alpha*grad)}")
    print(f"total iter = {(ep ) * m//r + cnt} , B = {B} , norm = {nrm}")
    

 
    return B  



#calculating expected loss
B_expected = np.array([[3,1,2]])
print(f"Thetha_expected = {B_expected}")
print(f"Training Loss - {loss_fun(X_train,Y_train,B_expected)}, Test Loss -{loss_fun(X_test,Y_test,B_expected)}, expected  - {B_expected}")



#call for different value of batch size (r) and loss calulation of train and test 
alpha = 0.001 #learinig rate
epoch = 100
r = 1
B_sgd = stoachstic_gradient_descent(r,alpha,epoch,X_train,Y_train)
loss_sgd_test  = loss_fun(X_test,Y_test,B_sgd)
loss_sgd_train = loss_fun(X_train,Y_train,B_sgd)
print(f"Batch Size = {r} Training Loss - {loss_sgd_train}, Test Loss -{loss_sgd_test}, Learned Parameters - {B_sgd}")


alpha = 0.001
epoch = 100
r = 80
B_sgd = stoachstic_gradient_descent(r,alpha,epoch,X_train,Y_train)
loss_sgd_test  = loss_fun(X_test,Y_test,B_sgd)
loss_sgd_train = loss_fun(X_train,Y_train,B_sgd)
print(f"Batch Size = {r}  Training Loss - {loss_sgd_train}, Test Loss -{loss_sgd_test}, Learned Parameters - {B_sgd}")

alpha = 0.001
epoch = 1000
r = 8000
B_sgd = stoachstic_gradient_descent(r,alpha,epoch,X_train,Y_train)
loss_sgd_test  = loss_fun(X_test,Y_test,B_sgd)
loss_sgd_train = loss_fun(X_train,Y_train,B_sgd)
print(f"Batch Size = {r}  Training Loss - {loss_sgd_train}, Test Loss -{loss_sgd_test}, Learned Parameters - {B_sgd}")

alpha = 0.001
epoch = 10000
r = 800000
B_sgd = stoachstic_gradient_descent(r,alpha,epoch,X_train,Y_train)
loss_sgd_test  = loss_fun(X_test,Y_test,B_sgd)
loss_sgd_train = loss_fun(X_train,Y_train,B_sgd)
print(f"Batch Size = {r}  Training Loss - {loss_sgd_train}, Test Loss -{loss_sgd_test}, Learned Parameters - {B_sgd}")


#CLOSED FORM SOLTUION CALCULATION 
def noraml_thetha(X,Y):
    T = X.T@X
    T_inv = np.linalg.inv(T)
    return T_inv @ (X.T @Y)

B_normal = noraml_thetha(X_train,Y_train).T
loss_normal_test = loss_fun(X_test,Y_test,B_normal)
loss_normal_train = loss_fun(X_train,Y_train,B_normal)

print(f"Training Loss - {loss_normal_train}, Test Loss -{loss_normal_test}, Learned Parameters(Closed form) - {B_normal}")



#Plots code 
def stoachstic_gradient_descent_plot(r,alpha,epoch,X,Y):
    idx = np.random.permutation(X_train.shape[0])
    X = X[idx]
    Y = Y[idx]
    B = np.zeros((1,X.shape[1]))
    

    m = len(X)
    delta = float('inf')
    cnt  = 0

    loss =  loss_fun(X[0:r ,: ],Y[0:r ,: ],B)

    B0 = [0]
    B1 = [0]
    B2 = [0]
    
    
    for ep in range(epoch):
       
        for cnt in range(m//r):

            start = (cnt * r) 
            end = min(start + r, m) 
            X_epoch = X[start:end, :]
            Y_epoch = Y[start:end, :]
            H_epoch = hypothesis(X_epoch,B)

            
            grad = np.dot((Y_epoch - H_epoch).T, X_epoch) / len(X_epoch)
            nrm = np.linalg.norm(alpha*grad) #gradient norm calcultion for convergence 
            
            if nrm < 1e-10 or int((ep ) * m//r + cnt) > 1e5  :
             
                print(f"total iter = {(ep ) * m//r + cnt} , B = {B} ,norm = {nrm} ")
                return np.array(B0),np.array(B1),np.array(B2)
            B = B + alpha * grad

            delta = loss - loss_fun(X_epoch,Y_epoch,B)
            loss = loss_fun(X_epoch,Y_epoch,B)
            
            cnt += 1 
           
            B0.append(B[0][0])
            B1.append(B[0][1])
            B2.append(B[0][2])
            

        # print(f"epoch = {ep}  , B = {B[0]} " , f"Loss = {loss}" ,f"Delta = {delta} , norm = {np.linalg.norm(alpha*grad)}")
    print(f"total iter = {(ep ) * m//r + cnt} , B = {B} , norm = {nrm}")
    return np.array(B0),np.array(B1),np.array(B2)

def sgd_plot(alpha , epoch , r ,X_train , Y_train ):
    

    B0,B1,B2 = stoachstic_gradient_descent_plot(r,alpha,epoch,X_train,Y_train)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(B0, B1, B2, color='blue', linewidth=2, label="Curve")

    ax.set_xlabel("θ0 axis")
    ax.set_ylabel("θ1 axis")
    ax.set_zlabel("θ2 axis")

    plt.show()


#plot call

alpha = 0.001
epoch = 100
r = 1
sgd_plot(alpha, epoch ,r ,X_train , Y_train)


alpha = 0.001
epoch = 100
r = 80
sgd_plot(alpha, epoch ,r ,X_train , Y_train)    

alpha = 0.001
epoch = 1000
r = 8000
sgd_plot(alpha, epoch ,r ,X_train , Y_train)

alpha = 0.001
epoch = 10000
r = 800000
sgd_plot(alpha, epoch ,r ,X_train , Y_train)



