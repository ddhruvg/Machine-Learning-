import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import tree
import sklearn
from sklearn.ensemble import RandomForestClassifier
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

def experiments():

    clf = sklearn.ensemble.RandomForestClassifier
    n_param = [50,150,250,350]
    mx_feature = [0.1,0.3,0.5,0.7]
    split_values = [2,4,6,8]
    res = []
    for n in n_param:
        for mx in mx_feature:
            for sp in split_values:
                clf =  RandomForestClassifier(n_estimators=n, max_features=mx, min_samples_split=sp, criterion='entropy')
                clf.fit(data_train.drop('result', axis=1), data_train['result'])
                train_acc = clf.score(data_train.drop('result', axis=1), data_train['result'])*100
                test_acc = clf.score(data_test.drop('result', axis=1), data_test['result'])*100
                val_acc = clf.score(data_validation.drop('result', axis=1), data_validation['result'])*100

                res.append([n,mx,sp,train_acc,test_acc,val_acc])
                print(f"n_estimators: {n}, max_features: {mx}, min_samples_split: {sp} => Train Acc: {train_acc} , Test Acc: {test_acc} , Val Acc: {val_acc}")
                

    best = sorted(res, key=lambda x: x[5], reverse=True)[0]
    print("Best Hyperparameters:")
    print(f"n_estimators: {best[0]}, max_features: {best[1]}, min_samples_split: {best[2]} => Train Acc: {best[3]} , Test Acc: {best[4]} , Val Acc: {best[5]}") 
    #out of the bag accuracy for best model 
    clf =  RandomForestClassifier(n_estimators=best[0], max_features=best[1], min_samples_split=best[2], criterion='entropy', oob_score=True)
    clf.fit(data_train.drop('result', axis=1), data_train['result'])
    print(f"OOB Accuracy: {clf.oob_score_*100} %")

best = [150,0.3,8]
clf =  RandomForestClassifier(n_estimators=best[0], max_features=best[1], min_samples_split=best[2], criterion='entropy', oob_score=True)
clf.fit(data_train.drop('result', axis=1), data_train['result'])
test_result = clf.predict(data_test.drop('result', axis=1))
result_to_data_frame = pd.DataFrame(test_result, columns=['result'])
result_to_data_frame.to_csv(OUTPUT_PATH, index=False)