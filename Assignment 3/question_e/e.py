import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import tree
import sys

TRAIN_PATH = sys.argv[1]
VALIDATION_PATH = sys.argv[2]
TEST_PATH = sys.argv[3]
OUTPUT_PATH = sys.argv[4]



data_train  = pd.read_csv(TRAIN_PATH)
data_test = pd.read_csv(TEST_PATH)
data_validation = pd.read_csv(VALIDATION_PATH)

categorical_cols = data_train.select_dtypes(include='object').columns.tolist()
data_train = pd.get_dummies(data_train, columns=categorical_cols ,  dtype=int)
data_validation = pd.get_dummies(data_validation, columns=categorical_cols ,  dtype=int)
data_test = pd.get_dummies(data_test, columns=categorical_cols ,  dtype=int)
data_test = data_test.reindex(columns=data_train.columns, fill_value=0)
data_validation = data_validation.reindex(columns=data_train.columns, fill_value=0)



def experimemnts():
    max_depth = [15,25,35,45]
    train_acc = []
    test_acc = []
    val_acc = []
    for dpt in max_depth: 
        clf  = tree.DecisionTreeClassifier(criterion ='entropy', max_depth = dpt )
        clf.fit(data_train.drop('result', axis=1), data_train['result'])
        train_acc.append(clf.score(data_train.drop('result', axis=1), data_train['result'])*100)
        test_acc.append(clf.score(data_test.drop('result', axis=1), data_test['result'])*100)
        val_acc.append(clf.score(data_validation.drop('result', axis=1), data_validation['result'])*100)
    plt.plot(max_depth , train_acc , label = "Train Accuracy ")
    plt.plot(max_depth , test_acc , label = "Test Accuracy ")
    plt.plot(max_depth , val_acc , label = "Validation Accuracy ")
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Max Depth for Decision Tree Classifier")
    plt.legend()
    plt.show()

    ccp_alpha_values = [0.0, 0.0001, 0.0003, 0.0005]
    train_acc = []
    test_acc = []
    val_acc = []
    for ccp  in ccp_alpha_values: 
        clf  = tree.DecisionTreeClassifier(criterion ='entropy',ccp_alpha=ccp )
        clf.fit(data_train.drop('result', axis=1), data_train['result'])
        train_acc.append(clf.score(data_train.drop('result', axis=1), data_train['result'])*100)
        test_acc.append(clf.score(data_test.drop('result', axis=1), data_test['result'])*100)
        val_acc.append(clf.score(data_validation.drop('result', axis=1), data_validation['result'])*100)
    plt.plot(ccp_alpha_values , train_acc , label = "Train Accuracy ")
    plt.plot(ccp_alpha_values , test_acc , label = "Test Accuracy ")
    plt.plot(ccp_alpha_values , val_acc , label = "Validation Accuracy ")
    plt.xlabel("CCP Alpha")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Max Depth for Decision Tree Classifier")
    plt.legend()
    plt.show()

clf  = tree.DecisionTreeClassifier(criterion ='entropy', max_depth = 35 , ccp_alpha=0.0005)
clf.fit(data_train.drop('result', axis=1), data_train['result'])
test_result = clf.predict(data_test.drop('result', axis=1))
result_to_data_frame = pd.DataFrame(test_result, columns=['result'])
result_to_data_frame.to_csv(OUTPUT_PATH, index=False)
