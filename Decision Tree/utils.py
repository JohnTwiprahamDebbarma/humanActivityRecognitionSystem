"""
Helper functions for the decision tree: check for real/discrete values, entropy, gini index,
variance and information gain.
"""

import pandas as pd
import numpy as np
from scipy.special import xlogy
from metrics import *


def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """

    if isinstance(y, int) or isinstance(y, float):
        return True

    if isinstance(y, pd.Series):
        if y.dtype in ("category", "object", "str"):
            return False
        
        return True
    
    return False

def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    val_counts = Y.value_counts()
    p = val_counts/val_counts.sum()
    entropy = -np.sum(xlogy(p, p)/np.log(2))

    return entropy


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    val_counts = Y.value_counts()
    p = val_counts/val_counts.sum()
    gini = 1 - np.sum(np.square(p))

    return gini


def variance(y: pd.Series) -> float:
    """
    Function to calculate the variance
    """
    return np.var(y)


def information(y: pd.Series, y_children: pd.Series, criterion=None) -> float:
    """
    Function to calculate the information gain
    """
    discrete = not check_ifreal(y)

    weights = [len(y_child)/len(y) for y_child in y_children]
    
    if discrete:
        if criterion in ("information_gain", None):
            info = entropy(y) - np.sum([w*entropy(y_child) for w, y_child in zip(weights, y_children)])
        elif criterion == "gini_index":
            info = gini_index(y) - np.sum([w*gini_index(y_child) for w, y_child in zip(weights, y_children)])
    else:
        info = mse(y) - np.sum([w*mse(y_child) for w, y_child in zip(weights, y_children)])
        # if criterion in ("information_gain", None):
        #     info = mse(y) - np.sum([w*mse(y_child) for w, y_child in zip(weights, y_children)])
        # elif criterion == "gini_index":
        #     info = variance(y) - np.sum([w*variance(y_child) for w, y_child in zip(weights, y_children)])

    return info

