from typing import Union
import pandas as pd
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """
    assert y_hat.size == y.size

    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    c=0
    for i in range(y.size):
        if y_hat.iloc[i]==y.iloc[i]:
            c+=1
    return c/y.size


def precision_help(y_hat, y, cls):
    """
    Function to calculate precision for a specific class
    """
    assert y_hat.size == y.size
    true_positives = sum((y_hat == cls) & (y == cls))
    all_predicted_positives = (y_hat == cls).sum()
    return true_positives / all_predicted_positives if all_predicted_positives > 0 else 0

def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision for the class cls
    """
    return precision_help(y_hat, y, cls)


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size
    c=0
    for i in range(y.size):
        if y.iloc[i]==y_hat.iloc[i] and y.iloc[i]==cls:
            c+=1
    return c/y.value_counts()[cls]


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size
    ms = mse(y_hat, y)
    return np.sqrt(ms)


def mse(y_hat: pd.Series, y: pd.Series=None) -> float:
    """
    Function to calculate the mean-squared-error(mse)
    """

    if y is None:
        y = y_hat.mean()

        # make sure y_hat and y have the same size
        y = np.array([y for _ in range(y_hat.size)])
        
    assert y_hat.size == y.size
    ms=np.square(y_hat-y).mean()
    return ms



def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    m_ae=np.abs(y_hat-y).mean()
    return m_ae