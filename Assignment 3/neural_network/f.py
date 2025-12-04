import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
import sys
from neural_network import NeuralNetwork 
import pickle

train_folder_path = sys.argv[1]
test_folder_path = sys.argv[2]
output_folder_path = sys.argv[3]



def get_data(type="train"):
    data = []
    label = []
    for i in range(37, 47):
        X = []
        y = []
        target = f"{i:02d}"  
        # train_data_path = f"{type}\{target}"
        train_data_path = os.path.join(type, target)
        
        for img_name in os.listdir(train_data_path):
            img_path = os.path.join(train_data_path, img_name)
            
            
            img = cv2.imread(img_path)
            
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            
            img = cv2.resize(img, (32, 32))
            img = img / 255.0 
            img_flatten = img.flatten()
            X.append(img_flatten)
            y.append(i)

        data.extend(X)
        label.extend(y)

    data = np.array(data)
    label = np.array(label)
    return data, label

X_train , y_train = get_data(train_folder_path)
X_test , y_test = get_data(test_folder_path)
y_train = y_train - 37
y_test = y_test - 37
# y_train  = np.eye(10)[y_train ].astype(np.float32)
# y_test  = np.eye(10)[y_test ].astype(np.float32)

test_predictions = []


hidden_layer_length = [512,256,128,64]
classes = np.unique(y_train)
train_F1 = []
test_F1  =[]
model = MLPClassifier(hidden_layer_sizes=hidden_layer_length,activation='relu',solver='sgd', alpha=0 , batch_size = 32,learning_rate='constant')
for i in range(20):
    model.partial_fit(X_train , y_train , classes = classes)
    if i == 19:
        y_test_pred = model.predict(X_test) + 37
        test_predictions.extend(y_test_pred)



with open(os.path.join(output_folder_path, 'mlp_weights.pkl'), 'rb') as f:
    params = pickle.load(f)

weights = params['weights']
biases = params['biases']


hidden_layer_length = [512,256,128,64]
classes = np.unique(y_train)
new_train_F1 = []
new_test_F1  = []
new_model = MLPClassifier(hidden_layer_sizes=hidden_layer_length,activation='relu',solver='sgd', alpha=0 , batch_size = 32,learning_rate='constant')
new_model.partial_fit(X_train[:1] , y_train[:1],classes = np.arange(10))
new_model.coefs_[:-1] = weights
new_model.intercepts_[:-1] = biases

for i in range(20):
    new_model.partial_fit(X_train , y_train , classes = np.arange(10))
    if  i == 19:
        y_test_pred = new_model.predict(X_test) + 37
        test_predictions.extend(y_test_pred)

df = pd.DataFrame(test_predictions, columns=['result'])
df.to_csv(os.path.join(output_folder_path, 'prediction_f.csv'), index=False)
