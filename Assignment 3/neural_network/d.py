import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.metrics import f1_score
import sys
from neural_network import NeuralNetwork 

train_folder_path = sys.argv[1]
test_folder_path = sys.argv[2]
output_folder_path = sys.argv[3]



def get_data(type="train"):
    data = []
    label = []
    for i in range(1, 37):
        X = []
        y = []
        target = f"{i:02d}"  
        # train_data_path = f"{type}/{target}"
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
y_train  = np.eye(36)[y_train - 1]
y_test  = np.eye(36)[y_test - 1]



hidden_layer_length = [[512],[512,256],[512,256,128],[512,256,128,64]]
test_predictions = []

for h_len in hidden_layer_length:
    model = NeuralNetwork(M = 32 , n = 3072 , HiddenLayer = h_len  , target_class = 36 , is_sigmoid = False , lr = 0.01) 
    model.fit(X_train , y_train , epochs = 20)
    y_test_prediction = model.predict(X_test)
    test_predictions.extend(y_test_prediction)
   
df = pd.DataFrame(test_predictions, columns=['result'])
df.to_csv(os.path.join(output_folder_path, 'prediction_d.csv'), index=False)
