"""
Decision Tree implemented from scratch.
This code is used to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from metrics import *

np.random.seed(42)


class Node():
    def __init__(self, feature_index=None, threshold=None, categories = None, children=None, info=None, value=None):
        ''' constructor ''' 
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.categories = categories
        self.children = children
        self.info = info
        
        # for leaf node (a decision node also stores it, for categories not seen while training)
        self.value = value


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion="information_gain", max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        self.root = self.build_tree(X, y)

    def get_leaf_value(self, y):
        ''' function to get the leaf value '''

        if check_ifreal(y):
            return y.mean()
        else:
            return y.value_counts().idxmax()

    def build_tree(self, X, y, depth=0):
        ''' recursive function to build the tree '''

        n_samples, n_features = X.shape
        n_labels = len(y.unique())
        
        if depth < self.max_depth and n_labels > 1:
            best_split = self.get_best_split(X, y)
            if best_split["info"] > 0:
                subtrees = []
                for X_child, y_child in zip(best_split["X_children"], best_split["y_children"]):
                    subtree = self.build_tree(X_child, y_child, depth+1)
                    subtrees.append(subtree)
            
                return Node(feature_index=best_split["feature_index"], threshold=best_split["threshold"], categories=best_split["categories"], children=subtrees, info=best_split["info"], value=self.get_leaf_value(y))

        leaf_value = self.get_leaf_value(y)
        return Node(value=leaf_value)
    
    def get_best_split(self, X, y):
        ''' function to find the best split '''

        best_split = {}
        best_split["feature_index"] = 0
        best_split["threshold"] = -np.inf
        best_split["categories"] = None
        best_split["X_children"] = None
        best_split["y_children"] = None
        best_split["info"] = -np.inf
        
        max_info = -np.inf
        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            feature_values = X.iloc[:, feature_index]
            dtype = feature_values.dtype
            possible_thresholds = pd.Series(feature_values.unique(), dtype=dtype)

            if len(possible_thresholds) == 1:
                continue

            discrete_feature = not check_ifreal(possible_thresholds)

            if discrete_feature:
                X_children, y_children = self.split(X, y, feature_index, None, discrete=True)

                # check if the split is valid
                if any([len(y_child) == 0 for y_child in y_children]):
                    continue

                info = information(y, y_children, self.criterion)

                if info > max_info:
                    best_split["feature_index"] = feature_index
                    best_split["threshold"] = None
                    best_split["categories"] = possible_thresholds.to_numpy()
                    best_split["X_children"] = X_children
                    best_split["y_children"] = y_children
                    best_split["info"] = info
                    max_info = info

            else:
                # sort possible thresholds
                possible_thresholds = np.sort(possible_thresholds)

                # get midpoints
                midpoints = (possible_thresholds[1:] + possible_thresholds[:-1])/2

                for threshold in midpoints:
                    X_children, y_children = self.split(X, y, feature_index, threshold, discrete=False)

                    # check if the split is valid
                    if any([len(y_child) == 0 for y_child in y_children]):
                        continue

                    info = information(y, y_children, self.criterion)

                    if info > max_info:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["categories"] = None
                        best_split["X_children"] = X_children
                        best_split["y_children"] = y_children
                        best_split["info"] = info
                        max_info = info

        return best_split
    
    def split(self, X, y, feature_index, threshold, discrete=False):
        ''' function to split the data '''

        if discrete:
            feature_values = X.iloc[:, feature_index].unique()

            X_children = []
            y_children = []

            for feature_value in feature_values:
                X_child = X[X.iloc[:, feature_index] == feature_value]
                y_child = y[X.iloc[:, feature_index] == feature_value]
                X_children.append(X_child)
                y_children.append(y_child)

            return X_children, y_children
        
        X_left = X[X.iloc[:, feature_index] <= threshold]
        X_right = X[X.iloc[:, feature_index] > threshold]
        y_left = y[X.iloc[:, feature_index] <= threshold]
        y_right = y[X.iloc[:, feature_index] > threshold]

        return (X_left, X_right), (y_left, y_right)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        y_hat = []
        for _, row in X.iterrows():
            y_hat.append(self.traverse_tree(self.root, row))

        return pd.Series(y_hat)
    
    def traverse_tree(self, node, row):
        ''' function to traverse the tree '''

        if node.children:
            if node.threshold is not None:
                if row.iloc[node.feature_index] <= node.threshold:
                    return self.traverse_tree(node.children[0], row)
                
                return self.traverse_tree(node.children[1], row)
            
            else:
                for i, category in enumerate(node.categories):
                    if row.iloc[node.feature_index] == category:
                        return self.traverse_tree(node.children[i], row)

                # category not seen while training: use the value stored at this node
                return node.value
        else:
            return node.value

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        self.print_tree(self.root, "")

    def print_tree(self, node, spacing=""):
        ''' function to print the tree '''

        if node.children:
            if node.threshold is not None:
                print(spacing + f"?(X{node.feature_index} <= {node.threshold})")
                for i, child in enumerate(node.children):
                    if i == 0:
                        print(spacing + "Y: ", end="")
                    else:
                        print(spacing + "N: ", end="")
                    self.print_tree(child, spacing + "\t")
            else:
                print(spacing + f"?(X{node.feature_index} in {node.categories})")
                for i, child in enumerate(node.children):
                    print(spacing + f"{node.categories[i]}: ", end="")
                    self.print_tree(child, spacing + "\t")
        else:
            print(spacing + f"Class/Value {node.value}")
