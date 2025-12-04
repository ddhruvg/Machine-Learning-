import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from collections import defaultdict
import math
import numpy as np
from PIL import Image
import glob
from cvxopt import matrix, solvers
import random
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import time 
import os
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from collections import defaultdict
import math
import numpy as np
from PIL import Image
import glob
from cvxopt import matrix, solvers
import random
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import time 
import os
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def get_data(type = "train"):
    path = f"data/{type}/ship"
    images = [img for img in os.listdir(path) ]
    processed_images = []

    for img in images:
        image = Image.open(os.path.join(path,img)).convert("RGB")
        img_array = np.array(image).reshape(-1)
        processed_images.append(img_array)

    X_ship = np.stack(processed_images)
    Y_ship = np.array([1 for _ in range(X_ship.shape[0])])

    path = f"data/{type}/truck"
    images = [img for img in os.listdir(path) ]
    processed_images = []

    for img in images:
        image = Image.open(os.path.join(path,img)).convert("RGB")
        img_array = np.array(image).reshape(-1)
        processed_images.append(img_array)

    X_truck = np.stack(processed_images)
    Y_truck = np.array([-1 for _ in range(X_truck.shape[0])])

    X = np.vstack((X_ship, X_truck))
    X = X/255.0
    Y = np.hstack((Y_ship, Y_truck))

    return X, Y

import cvxopt
import numpy as np

class SupportVectorMachine:
    '''
    Binary Classifier using Support Vector Machine
    '''
    def __init__(self):
        self.W = None
        self.b = None
        self.alpha = None
        self.kernel = None
        self.X_sv = None
        self.Y_sv = None
        self.alpha_sv = None
        self.alpha = None
        self.C = None
        self.gamma = None
        pass
    def linear_kernel(self, x1, x2):
        K = x1 @ x2.T
        return K
    def gaussian_kernel(self,x1,x2,gamma =0.001):
        m1 = x1.shape[0]
        m2 = x2.shape[0]
        x1_sq = np.sum(x1**2, axis=1).reshape((m1,1))
        x2_sq = np.sum(x2**2, axis=1).reshape((1,m2))
        k  =np.exp(-gamma * (x1_sq + x2_sq - 2 * (x1 @ x2.T)))
        return k
            
    def fit(self, X, y, kernel = 'linear', C = 1.0, gamma = 0.001):
        '''
        Learn the parameters from the given training data
        Classes are 0 or 1
        
        Args:
            X: np.array of shape (N, D) 
                where N is the number of samples and D is the flattened dimension of each image
                
            y: np.array of shape (N,)
                where N is the number of samples and y[i] is the class of the ith sample
                
            kernel: str
                The kernel to be used. Can be 'linear' or 'gaussian'
                
            C: float
                The regularization parameter
                
            gamma: float
                The gamma parameter for gaussian kernel, ignored for linear kernel
        '''
        self.kernel = kernel 
        self.C = C
        self.gamma = gamma
        Y = y
        t1 = time.time()
        m, n = X.shape
        K = self.linear_kernel(X, X) if kernel == 'linear' else self.gaussian_kernel(X, X, gamma=gamma)
        P = matrix(np.outer(Y, Y) * K)
        q = matrix(-np.ones(m)) 
        G = matrix(np.vstack((-np.eye(m), np.eye(m))))
        h = matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
        A = matrix(Y, (1, m), 'd')
        b = matrix(0.0)
        

        sol = solvers.qp(P, q, G, h, A, b)  
        alpha = np.ravel(sol['x'])

        if kernel == 'linear':
            w = ((alpha * Y) @ X)

            

            wTx_neg = X[Y == -1] @ w
            wTx_pos = X[Y == 1] @ w

            b = - (np.max(wTx_neg) + np.min(wTx_pos)) / 2
            t2 = time.time() - t1    
            
            self.W = w
            self.b = b
            self.alpha = alpha 
        else:
            sv = alpha > 1e-5
            sv_index = np.where(sv)[0]
            print(f"number of support vectors: {len(sv_index)}")
            print(f"percentage of support vectors: {(len(sv_index)/m)*100}%")
            X_sv = X[sv]
            y_sv = Y[sv]
            alpha_sv = alpha[sv]
            t2 = time.time() - t1
            print(f"Time taken to train SVM gaussian using cvxopt: {t2} seconds")
            self.X_sv = X_sv
            self.Y_sv = y_sv
            self.alpha_sv = alpha_sv
            self.alpha = alpha
            

        

    def predict(self, X):
        '''
        Predict the class of the input data
        
        Args:
            X: np.array of shape (N, D) 
                where N is the number of samples and D is the flattened dimension of each image
                
        Returns:
            np.array of shape (N,)
                where N is the number of samples and y[i] is the class of the
                ith sample (0 or 1)
        '''
        if self.kernel == 'linear':
            ans = np.sign(X @ self.W + self.b)
        else:    
            sep_boundary = (self.alpha_sv > 1e-5) & (self.alpha_sv < self.C)
            K_boundary = self.gaussian_kernel( self.X_sv[sep_boundary],self.X_sv , self.gamma)
            b = self.Y_sv[sep_boundary] - np.sum(self.alpha_sv * self.Y_sv * K_boundary, axis=1)
            b = np.mean(b)   

            k_test = self.gaussian_kernel(X, self.X_sv, self.gamma)
            ans = k_test @ (self.alpha_sv * self.Y_sv) + b
            ans = np.sign(ans)

        ans[ans == -1] = 0
        return ans       
        
def calcualte_matrices(X,Y,C=1.0):
    t1 = time.time()
    m, n = X.shape
    K = X @ X.T
    P = matrix(np.outer(Y, Y) * K)
    q = matrix(-np.ones(m)) 
    G = matrix(np.vstack((-np.eye(m), np.eye(m))))
    h = matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = matrix(Y, (1, m), 'd')
    b = matrix(0.0)
    

    sol = solvers.qp(P, q, G, h, A, b)  
    alpha = np.ravel(sol['x'])
    w = ((alpha * Y) @ X)

    

    wTx_neg = X[Y == -1] @ w
    wTx_pos = X[Y == 1] @ w

    b = - (np.max(wTx_neg) + np.min(wTx_pos)) / 2
    t2 = time.time() - t1    
    print(f"Time taken to train SVM linear using cvxopt: {t2} seconds")
    return w,b,alpha

def predict(X, w, b):
    return np.sign(X @ w + b)

X_train , Y_train = get_data("train")
X_test , Y_test = get_data("test")    

W_cvx_linear,b_cvx_linear,alpha_linear = calcualte_matrices(X_train,Y_train,C=1.0)
y_predictions_train = predict(X_train, W_cvx_linear, b_cvx_linear)
y_predictions_test = predict(X_test, W_cvx_linear, b_cvx_linear)

acc = accuracy_score(Y_test, y_predictions_test)
print("Test Accuracy:", acc)

acc = accuracy_score(Y_train, y_predictions_train)
print("Train Accuracy:", acc)

svm_ = SupportVectorMachine()
svm_.fit(X_train, Y_train, kernel='gaussian', C=1.0,gamma=0.001)

predict_ = svm_.predict(X_test)
Y_test_ = Y_test
Y_test_[Y_test_ == -1] = 0 
acc = accuracy_score(Y_test_, predict_)
print(acc)

sv = alpha_linear > 1e-5
number_0f_sv = np.sum(sv)
percent_sv = (number_0f_sv/len(Y_train))*100
print("Number of Support Vectors:", number_0f_sv)
print("Percentage of Support Vectors:", percent_sv)

sv_index = np.where(sv)[0]
sv_alpha = alpha_linear[sv]
top5_index = sv_index[np.argsort(-sv_alpha)[:5]]
top5_images = X_train[top5_index].reshape(-1,32,32,3)
w_images = W_cvx_linear.reshape(32,32,3)

plt.figure(figsize=(15,3))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(top5_images[i])
    plt.axis('off')
    plt.title(f"SV{i+1}")
plt.show() 

plt.figure(figsize=(4,4))
plt.imshow(w_images - w_images.min())  # normalize for visualization
plt.axis('off')
plt.title("Weight vector w")
plt.show()

def gaussian_kernel(x1,x2,gamma =0.001):
    m1 = x1.shape[0]
    m2 = x2.shape[0]
    x1_sq = np.sum(x1**2, axis=1).reshape((m1,1))
    x2_sq = np.sum(x2**2, axis=1).reshape((1,m2))
    k  =np.exp(-gamma * (x1_sq + x2_sq - 2 * (x1 @ x2.T)))
    return k
def calcualte_matrices_gaussian(X,Y,C=1.0):
    t1 = time.time()
    m, n = X.shape
    K = gaussian_kernel(X,X,gamma=0.001)
    P = matrix(np.outer(Y, Y) * K)
    q = matrix(-np.ones(m)) 
    G = matrix(np.vstack((-np.eye(m), np.eye(m))))
    h = matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = matrix(Y, (1, m), 'd')
    b = matrix(0.0)
    

    sol = solvers.qp(P, q, G, h, A, b)  
    alpha = np.ravel(sol['x'])
   

    sv = alpha > 1e-5
    sv_index = np.where(sv)[0]
    print(f"number of support vectors: {len(sv_index)}")
    print(f"percentage of support vectors: {(len(sv_index)/m)*100}%")
    X_sv = X[sv]
    y_sv = Y[sv]
    alpha_sv = alpha[sv]
    t2 = time.time() - t1
    print(f"Time taken to train SVM gaussian using cvxopt: {t2} seconds")
    return X_sv , y_sv ,alpha_sv, alpha 
   


def predict_gaussian(X_sv, y_sv, alpha_sv , X_test ,gamma = 0.001 ,C = 1.0):
    sep_boundary = (alpha_sv > 1e-5) & (alpha_sv < C)
    K_boundary = gaussian_kernel( X_sv[sep_boundary],X_sv , gamma)
    b = y_sv[sep_boundary] - np.sum(alpha_sv * y_sv * K_boundary, axis=1)
    b = np.mean(b)   

    k_test = gaussian_kernel(X_test, X_sv, gamma)
    y_predict = k_test @ (alpha_sv * y_sv) + b 
    print(y_predict)
    return np.sign(y_predict)
       

X_sv, y_sv, alpha_sv, alpha_gaussian = calcualte_matrices_gaussian(X_train, Y_train, C=1.0)

y_pred = predict_gaussian(X_sv, y_sv, alpha_sv, X_test, gamma=0.001, C=1.0)


accuracy = np.mean(y_pred == Y_test)
print("Test Accuracy:", accuracy)

sv_linear_cvxopt = np.where(alpha_linear > 1e-5)[0]
sv_gaussian_cvxopt = np.where(alpha_gaussian > 1e-5)[0]
X_sv_linear = X_train[sv_linear_cvxopt]
X_sv_gaussian = X_train[sv_gaussian_cvxopt]



match = 0
for x in X_sv_linear:
    if   np.any(np.linalg.norm(X_sv_gaussian - x, axis=1) < 1e-5):
        match += 1

print("Number of matching support vectors:", match)
print("percentage of matching support vectors:", (match/len(sv_linear_cvxopt))*100)


sv_gaussian = alpha_gaussian > 1e-5
sv_index_gaussian  = np.where(sv_gaussian)[0]
sv_alpha_gaussian = alpha_gaussian[sv_gaussian]
top5_index_gaussian = sv_index[np.argsort(-sv_alpha)[:5]]
top5_images_gaussian = X_train[top5_index_gaussian].reshape(-1,32,32,3)
plt.figure(figsize=(15,3))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(top5_images_gaussian[i])
    plt.axis('off')
    plt.title(f"SV{i+1}")
plt.show() 

sv_linear_cvxopt = np.where(alpha_linear > 1e-5)[0]
sv_gaussian_cvxopt = np.where(alpha_gaussian > 1e-5)[0]
X_sv_linear = X_train[sv_linear_cvxopt]
X_sv_gaussian = X_train[sv_gaussian_cvxopt]


t1 = time.time()
linear_svm = SVC(kernel='linear', C=1.0)
linear_svm.fit(X_train, Y_train)
t2 = time.time()
print("Time taken by LIBSVM for linear kernel:", t2 - t1)
sv_linear_svm = linear_svm.support_
print("Number of support vectors by LIBSVM (linear kernel):", len(sv_linear_svm))

t1 = time.time()
gaussian_svm = SVC(kernel='rbf', C=1.0, gamma=0.001)
gaussian_svm.fit(X_train, Y_train)
t2 = time.time()
print("Time taken by LIBSVM for Gaussian kernel:", t2 - t1)
sv_gaussian_svm = gaussian_svm.support_
print("Number of support vectors by LIBSVM (Gaussian kernel):", len(sv_gaussian_svm))

print(f"number of support vector for linear kernel using cvxopt: {len(sv_linear_cvxopt)}")
print(f"number of support vector for gaussian kernel using cvxopt: {len(sv_gaussian_cvxopt)}")

def overlap(X_sv_cvx, X_sv_sklearn, tol=1e-5):
    matches = 0
    for x in X_sv_cvx:
        if np.any(np.linalg.norm(X_sv_sklearn - x, axis=1) < tol):
            matches += 1
    return matches, matches / len(X_sv_cvx)

matches_linear, frac_linear = overlap(X_sv_linear, X_train[sv_linear_svm])
matches_gaussian, frac_gaussian = overlap(X_sv_gaussian, X_train[sv_gaussian_svm])

print(f"Matching SVs (linear): {matches_linear}/{len(X_sv_linear)} ({frac_linear*100}%)")
print(f"Matching SVs (Gaussian): {matches_gaussian}/{len(X_sv_gaussian)} ({frac_gaussian*100}%)")

w_sklearn_linear = linear_svm.coef_.ravel()
b_sklearn_linear = linear_svm.intercept_[0]

print(f"W for sklearn linear svm is {w_sklearn_linear} , and b for sklearn linear svm is {b_sklearn_linear}")
print(f"W for cvxopt linear svm is {W_cvx_linear} , and b for cvxopt linear svm is {b_cvx_linear}")

w_img = w_sklearn_linear.reshape(32,32,3)
plt.figure(figsize=(4,4))
plt.imshow(w_img - w_img.min())  # normalize for visualization
plt.axis('off')
plt.title("Weight vector w")
plt.show()

print(f"Difference in W (linear): {np.linalg.norm(w_sklearn_linear - W_cvx_linear)}")

Y_prediction_sklear_linear = linear_svm.predict(X_test)
Y_predictions_sklearn_gaussian = gaussian_svm.predict(X_test)
accuracy_sklearn_linear = accuracy_score(Y_test, Y_prediction_sklear_linear)
accuracy_sklearn_gaussian = accuracy_score(Y_test, Y_predictions_sklearn_gaussian)
print(f"Accuracy for sklearn linear svm is {accuracy_sklearn_linear * 100}")
print(f"Accuracy for sklearn gaussian svm is {accuracy_sklearn_gaussian * 100}")

def gaussian_kernel(x1,x2,gamma =0.001):
    m1 = x1.shape[0]
    m2 = x2.shape[0]
    x1_sq = np.sum(x1**2, axis=1).reshape((m1,1))
    x2_sq = np.sum(x2**2, axis=1).reshape((1,m2))
    k  =np.exp(-gamma * (x1_sq + x2_sq - 2 * (x1 @ x2.T)))
    return k
def calcualte_matrices_gaussian(X,Y,C=1.0):
    t1 = time.time()
    m, n = X.shape
    K = gaussian_kernel(X,X,gamma=0.001)
    P = matrix(np.outer(Y, Y) * K)
    q = matrix(-np.ones(m)) 
    G = matrix(np.vstack((-np.eye(m), np.eye(m))))
    h = matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = matrix(Y, (1, m), 'd')
    b = matrix(0.0)
    

    sol = solvers.qp(P, q, G, h, A, b)  
    alpha = np.ravel(sol['x'])
   

    sv = alpha > 1e-5
    sv_index = np.where(sv)[0]
    print(f"number of support vectors: {len(sv_index)}")
    print(f"percentage of support vectors: {(len(sv_index)/m)*100}%")
    X_sv = X[sv]
    y_sv = Y[sv]
    alpha_sv = alpha[sv]
    t2 = time.time() - t1
    print(f"Time taken to train SVM gaussian using cvxopt: {t2} seconds")
    return X_sv , y_sv ,alpha_sv, alpha 
   


def predict_gaussian(X_sv, y_sv, alpha_sv , X_test ,gamma = 0.001 ,C = 1.0):
    sep_boundary = (alpha_sv > 1e-5) & (alpha_sv < C)
    K_boundary = gaussian_kernel( X_sv[sep_boundary],X_sv , gamma)
    b = y_sv[sep_boundary] - np.sum(alpha_sv * y_sv * K_boundary, axis=1)
    b = np.mean(b)   

    k_test = gaussian_kernel(X_test, X_sv, gamma)
    y_predict = k_test @ (alpha_sv * y_sv) + b 
    return np.sign(y_predict)
    






def gaussian_kernel(x1,x2,gamma =0.001):
    m1 = x1.shape[0]
    m2 = x2.shape[0]
    x1_sq = np.sum(x1**2, axis=1).reshape((m1,1))
    x2_sq = np.sum(x2**2, axis=1).reshape((1,m2))
    k  =np.exp(-gamma * (x1_sq + x2_sq - 2 * (x1 @ x2.T)))
    return k
def calcualte_matrices_gaussian(X,Y,C=1.0):
    t1 = time.time()
    m, n = X.shape
    K = gaussian_kernel(X,X,gamma=0.001)
    P = matrix(np.outer(Y, Y) * K)
    q = matrix(-np.ones(m)) 
    G = matrix(np.vstack((-np.eye(m), np.eye(m))))
    h = matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = matrix(Y, (1, m), 'd')
    b = matrix(0.0)
    
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)  
    alpha = np.ravel(sol['x'])
   

    sv = alpha > 1e-5
    sv_index = np.where(sv)[0]
    print(f"number of support vectors: {len(sv_index)}")
    print(f"percentage of support vectors: {(len(sv_index)/m)*100}%")
    X_sv = X[sv]
    y_sv = Y[sv]
    alpha_sv = alpha[sv]
    t2 = time.time() - t1
    print(f"Time taken to train SVM gaussian using cvxopt: {t2} seconds")
    return X_sv , y_sv ,alpha_sv, alpha 
   


def predict_gaussian(X_sv, y_sv, alpha_sv , X_test ,gamma = 0.001 ,C = 1.0):
    sep_boundary = (alpha_sv > 1e-5) & (alpha_sv < C)
    K_boundary = gaussian_kernel( X_sv[sep_boundary],X_sv , gamma)
    b = y_sv[sep_boundary] - np.sum(alpha_sv * y_sv * K_boundary, axis=1)
    b = np.mean(b)   

    k_test = gaussian_kernel(X_test, X_sv, gamma)
    y_predict = k_test @ (alpha_sv * y_sv) + b 
    return np.sign(y_predict), np.array(y_predict)
       


def get_data(type = "train", labels = ["ship" , "truck"]):
    path = f"data/{type}/{labels[0]}"
    images = [img for img in os.listdir(path) ]
    processed_images = []

    for img in images:
        image = Image.open(os.path.join(path,img)).convert("RGB")
        img_array = np.array(image).reshape(-1)
        processed_images.append(img_array)

    X_ship = np.stack(processed_images)
    Y_ship = np.array([1 for _ in range(X_ship.shape[0])])

    path = f"data/{type}/{labels[1]}"
    images = [img for img in os.listdir(path) ]
    processed_images = []

    for img in images:
        image = Image.open(os.path.join(path,img)).convert("RGB")
        img_array = np.array(image).reshape(-1)
        processed_images.append(img_array)

    X_truck = np.stack(processed_images)
    Y_truck = np.array([-1 for _ in range(X_truck.shape[0])])

    X = np.vstack((X_ship, X_truck))
    X = X/255.0
    Y = np.hstack((Y_ship, Y_truck))

    return X, Y


list_of_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

        

x_store = []
y_store = []

for i in range(len(list_of_labels)):
    label = list_of_labels[i]
    path = f"data/test/{label}"
    images = [img for img in os.listdir(path) ]
    processed_images = []

    for img in images:
        image = Image.open(os.path.join(path,img)).convert("RGB")
        img_array = np.array(image).reshape(-1)
        processed_images.append(img_array)

    X_ship = np.stack(processed_images)
    Y_ship = np.array([i for _ in range(X_ship.shape[0])])
    x_store.append(X_ship)
    y_store.append(Y_ship)

X_test_multi = np.vstack(x_store)/255.0
Y_test_multi = np.hstack(y_store)


t1 = time.time()
clssifier_pair = {}

for i in range(len(list_of_labels)-1):
    for j in range(i+1, len(list_of_labels)):
        label_pair = (list_of_labels[i], list_of_labels[j])

        print(f"Training classifier for labels: {label_pair}")
        X_train, Y_train = get_data("train", labels=list(label_pair))
        
        X_sv, y_sv, alpha_sv, alpha_gaussian = calcualte_matrices_gaussian(X_train, Y_train, C=1.0)
        clssifier_pair[label_pair] = (X_sv, y_sv, alpha_sv)
print(f"Total time taken to train all classifiers: {time.time() - t1} seconds")     
        

y_predict_pair = {}
for key in clssifier_pair.keys():
    label_pair = key
    X_sv, y_sv, alpha_sv = clssifier_pair[key]
    y_predict_temp,score  = predict_gaussian(X_sv, y_sv, alpha_sv, X_test_multi, gamma=0.001, C=1.0)
    y_predict_pair[label_pair] = (y_predict_temp,score)


label_map = {label: idx for idx, label in enumerate(list_of_labels)}
final_prediction = []

for i in range(X_test_multi.shape[0]):
    votes = np.zeros(len(list_of_labels))
    scores = np.zeros(len(list_of_labels))
    for key in y_predict_pair.keys():
        label1, label2 = key
        pred, score = y_predict_pair[key][0][i], y_predict_pair[key][1][i]
        if pred == 1:
            votes[label_map[label1]] += 1
            scores[label_map[label1]] += abs(score)
        else:
            votes[label_map[label2]] += 1
            scores[label_map[label2]] += abs(score)
    tied_labels = np.where(votes == np.max(votes))[0]
    if len(tied_labels) > 1:
        final_label = tied_labels[np.argmax(scores[tied_labels])]
        final_prediction.append(final_label)
    else:            
        final_label = np.argmax(votes)
        final_prediction.append(final_label) 
final_prediction = np.array(final_prediction)
accuracy = accuracy_score(Y_test_multi, final_prediction)
print(f"Final accuracy using One-vs-One SVM with Gaussian kernel: {accuracy*100}%")         
              
        
X_train_multi = []
Y_train_multi = []

for i in range(len(list_of_labels)):
    label = list_of_labels[i]
    path = f"data/train/{label}"
    images = [img for img in os.listdir(path) ]
    processed_images = []

    for img in images:
        image = Image.open(os.path.join(path,img)).convert("RGB")
        img_array = np.array(image).reshape(-1)
        processed_images.append(img_array)

    X_ship = np.stack(processed_images)
    Y_ship = np.array([i for _ in range(X_ship.shape[0])])
    X_train_multi.append(X_ship)
    Y_train_multi.append(Y_ship)

X_train_multi = np.vstack(X_train_multi)/255.0
Y_train_multi = np.hstack(Y_train_multi)

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


start = time.time()
multi_svm = svm.SVC(kernel='rbf', C=1.0, gamma=0.001 )
multi_svm.fit(X_train_multi, Y_train_multi)
train_time = time.time() - start
print(f"Time taken to train multi-class SVM using sklearn: {train_time} seconds")
y_prediction_sklearn = multi_svm.predict(X_test_multi)
accuracy_sklearn = accuracy_score(Y_test_multi, y_prediction_sklearn)
print(f"Final accuracy using sklearn SVM with Gaussian kernel: {accuracy_sklearn*100}%") 

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Compute confusion matrix
cm = confusion_matrix(Y_test_multi,y_prediction_sklearn)
labels = list(range(10))

# Create a bigger, clearer plot
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True,          # display numbers
    fmt='d',             # integer format
    cmap='Blues',        # color map
    xticklabels=labels, 
    yticklabels=labels,
    cbar=True,
    annot_kws={"size": 9}  # adjust font size for annotations
)

plt.title("Confusion Matrix", fontsize=16, pad=15)
plt.xlabel("Predicted Class", fontsize=14)
plt.ylabel("True Class", fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Compute confusion matrix
cm = confusion_matrix(Y_test_multi,final_prediction)
labels = list(range(10))

# Create a bigger, clearer plot
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True,          # display numbers
    fmt='d',             # integer format
    cmap='Blues',        # color map
    xticklabels=labels, 
    yticklabels=labels,
    cbar=True,
    annot_kws={"size": 9}  # adjust font size for annotations
)

plt.title("Confusion Matrix", fontsize=16, pad=15)
plt.xlabel("Predicted Class", fontsize=14)
plt.ylabel("True Class", fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()


wrong_prediction = np.zeros(10)
for i in range(len(Y_test_multi)):
    if Y_test_multi[i] != final_prediction[i]:
        wrong_prediction[Y_test_multi[i]] += 1

max_wrong_predicted_class = list_of_labels[np.argmax(wrong_prediction)]
print(f"Class with maximum wrong predictions for cvxopt: {max_wrong_predicted_class}")

wrong_prediction = np.zeros(10)
for i in range(len(Y_test_multi)):
    if Y_test_multi[i] != y_prediction_sklearn[i]:
        wrong_prediction[Y_test_multi[i]] += 1

max_wrong_predicted_class = list_of_labels[np.argmax(wrong_prediction)]
print(f"Class with maximum wrong predictions for sklearn: {max_wrong_predicted_class}")

index = []
for i in range(len(Y_test_multi)):
    if Y_test_multi[i] != final_prediction[i]:
        index.append(i)
    if len(index) == 10:
        break
X_test_misclassified = X_test_multi[index].reshape(-1,32,32,3)
plt.figure(figsize=(15,3))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(X_test_misclassified[i])
    plt.axis('off')
    plt.title(f"SV{i+1}")
plt.show() 
   

index = []
for i in range(len(Y_test_multi)):
    if Y_test_multi[i] != y_prediction_sklearn[i]:
        index.append(i)
    if len(index) == 10:
        break
X_test_misclassified = X_test_multi[index].reshape(-1,32,32,3)
plt.figure(figsize=(15,3))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(X_test_misclassified[i])
    plt.axis('off')
    plt.title(f"SV{i+1}")
plt.show() 


def f(X_train_multi , Y_train_multi, X_test_multi, Y_test_multi,c):
 
    multi_svm = svm.SVC(kernel='rbf',C=c , gamma=0.001 )
    multi_svm.fit(X_train_multi, Y_train_multi)
    
    y_prediction_sklearn = multi_svm.predict(X_test_multi)
    accuracy_sklearn = accuracy_score(Y_test_multi, y_prediction_sklearn)
    return accuracy_sklearn
   

idx= np.arange(X_train_multi.shape[0])
np.random.shuffle(idx)
X = X_train_multi[idx]
Y = Y_train_multi[idx]

splits = np.array_split(idx,5)

acc_score = []
test_acc = []

c_values = [10**-5, 10**-3, 1, 5, 10]\

def cv_scratch(X, Y, splits, c_values):

    for c in c_values:
        acc = []
        for arr in splits:
            X_test , Y_test = X[arr], Y[arr]
            mask = np.ones(len(X), dtype=bool)
            mask[arr] = False
            X_train, Y_train = X[mask], Y[mask]
            acc.append(f(X_train, Y_train, X_test, Y_test,c))

        acc_score.append(np.mean(acc))
        test_acc.append(f(X, Y, X_test_multi, Y_test_multi,c))

    print(acc_score)
    print(test_acc)


    plt.figure(figsize=(8, 5))
    plt.plot(c_values, acc_score, '-o', label='CV')
    plt.plot(c_values, test_acc, '-o', label='test')
    plt.xscale('log',fontsize=12)
    plt.xlabel('C values',fontsize = 12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()  



X_train = X_train_multi
y_train = Y_train_multi
X_test = X_test_multi
y_test = Y_test_multi

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score


C_values = [1e-5, 1e-3, 1, 5, 10]
gamma = 0.001  # Fixed as per problem
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []
test_scores = []

for C in C_values:
   
    svm = SVC(kernel='rbf', gamma=gamma, C=C)

    
    scores = cross_val_score(svm, X_train, y_train, cv=kf)
    cv_mean = np.mean(scores)
    cv_scores.append(cv_mean)

    
    svm.fit(X_train, y_train)
    test_acc = svm.score(X_test, y_test)
    test_scores.append(test_acc)

    print(f"C={C}: CV Accuracy={cv_mean*100:.2f}%, Test Accuracy={test_acc*100:.2f}%")


plt.figure(figsize=(8,6))
plt.plot(C_values, cv_scores, marker='o', label='5-Fold CV Accuracy')
plt.plot(C_values, test_scores, marker='s', label='Test Accuracy')
plt.xscale('log')
plt.xlabel('C (log scale)')
plt.ylabel('Accuracy')
plt.title('SVM Gaussian Kernel: CV vs Test Accuracy')
plt.grid(True)
plt.legend()
plt.show()


best_idx = np.argmax(cv_scores)
best_C = C_values[best_idx]
print(f"\nBest C based on 5-fold CV: {best_C}")

final_svm = SVC(kernel='rbf', gamma=gamma, C=best_C)
final_svm.fit(X_train, y_train)
final_test_acc = final_svm.score(X_test, y_test)
print(f"Final Test Accuracy with best C={best_C}: {final_test_acc*100:.2f}%")























