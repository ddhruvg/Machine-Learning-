import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys

TRAIN_PATH = sys.argv[1]
VALIDATION_PATH = sys.argv[2]
TEST_PATH = sys.argv[3]
OUTPUT_PATH = sys.argv[4]



data_train  = pd.read_csv(TRAIN_PATH)
data_test = pd.read_csv(TEST_PATH)
data_validation = pd.read_csv(VALIDATION_PATH)

categorical_cols = data_train.select_dtypes(include='object').columns.tolist()

data_train = pd.get_dummies(data_train, columns=categorical_cols, dtype=int)
data_validation = pd.get_dummies(data_validation, columns=categorical_cols, dtype=int)
data_test = pd.get_dummies(data_test, columns=categorical_cols, dtype=int)

data_test = data_test.reindex(columns=data_train.columns, fill_value=0)
data_validation = data_validation.reindex(columns=data_train.columns, fill_value=0)


X_train = data_train.drop('result', axis=1).values
y_train = data_train['result'].values

X_validation = data_validation.drop('result', axis=1).values
y_validation = data_validation['result'].values

X_test = data_test.drop('result', axis=1).values
y_test = data_test['result'].values

class Node:
    def __init__(self, feature=None, threshold=None, is_leaf=False, target=None):
        self.feature = feature
        self.threshold = threshold
        self.is_leaf = is_leaf
        self.target = target
        self.left = None
        self.right = None


class DecisionTree:
    def __init__(self, max_depth=5, X=None, y=None, feature_names=None):
        self.max_depth = max_depth
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.root = None

    def calculate_entropy(self, y):
        if len(y) == 0:
            return 0
        p = np.mean(y == 1)
        if p in [0, 1]:
            return 0
        return - (p * math.log2(p) + (1 - p) * math.log2(1 - p))

    def calculate_information_gain(self, X, y, feature_index):
        threshold = np.median(X[:, feature_index])
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold

        y_left = y[left_mask]
        y_right = y[right_mask]

        H = (len(y_left)/len(y)) * self.calculate_entropy(y_left) + \
            (len(y_right)/len(y)) * self.calculate_entropy(y_right)
        return self.calculate_entropy(y) - H

    def get_best_split(self, X, y):
        best_gain = 0
        best_feature = None
        for feature_index in range(X.shape[1]):
            gain = self.calculate_information_gain(X, y, feature_index)
            if gain >= best_gain:
                best_gain = gain
                best_feature = feature_index
        return best_feature

    def build_tree(self, X, y, depth):
        if depth == self.max_depth or self.calculate_entropy(y) == 0:
            target = 1 if np.sum(y == 1) >= np.sum(y == 0) else 0
            return Node(is_leaf=True, target=target)

        best_feature = self.get_best_split(X, y)
        root = Node(feature=best_feature)
        root.target = 1 if np.sum(y == 1) >= np.sum(y == 0) else 0

        threshold = np.median(X[:, best_feature])
        left_mask = X[:, best_feature] <= threshold
        right_mask = X[:, best_feature] > threshold

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        if len(y_left) == 0:
            target = 1 if np.sum(y == 1) >= np.sum(y == 0) else 0
            child_left = Node(is_leaf=True, target=target)
        else:
            child_left = self.build_tree(X_left, y_left, depth + 1)

        if len(y_right) == 0:
            target = 1 if np.sum(y == 1) >= np.sum(y == 0) else 0
            child_right = Node(is_leaf=True, target=target)
        else:
            child_right = self.build_tree(X_right, y_right, depth + 1)

        root.left = child_left
        root.right = child_right
        root.threshold = threshold
        return root

    def fit(self):
        self.root = self.build_tree(self.X, self.y, 0)

    def predict(self, X):
        y_predictions = []
        for i in range(X.shape[0]):
            current_node = self.root
            while not current_node.is_leaf:
                feature = current_node.feature
                threshold = current_node.threshold
                if X[i, feature] <= threshold:
                    current_node = current_node.left
                else:
                    current_node = current_node.right
            y_predictions.append(current_node.target)
        return np.array(y_predictions)
    

def experimemnts():
    depth = [15, 25 , 35 , 45]
    for dpt in depth:
        

        Dtree = DecisionTree(max_depth=dpt, X=X_train, y=y_train)
        Dtree.fit()

        
        validation_result = Dtree.predict(X_validation)
        test_result = Dtree.predict(X_test)
        train_result = Dtree.predict(X_train)

        
        correct_predictions_validation = np.sum(validation_result == y_validation)
        correct_predictions_test = np.sum(test_result == y_test)
        correct_predictions_train = np.sum(train_result == y_train)

        print(f"Train accuracy for depth {dpt}: {correct_predictions_train / len(y_train) * 100:.2f}%")
        print(f"Validation accuracy for depth {dpt}: {correct_predictions_validation / len(y_validation) * 100:.2f}%")
        print(f"Test accuracy for depth {dpt}: {correct_predictions_test / len(y_test) * 100:.2f}%")



Dtree = DecisionTree(max_depth=45, X=X_train, y=y_train)
Dtree.fit()
test_result = Dtree.predict(X_test)
result_to_data_frame = pd.DataFrame(test_result, columns=['result'])
result_to_data_frame.to_csv(OUTPUT_PATH, index=False)

