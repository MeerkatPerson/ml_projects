# -*- coding: utf-8 -*-
import numpy as np


def sigmoid(t):
    """Compute the sigmoid function on t."""
    return 1./ (1+ np.exp(-t))

def GD_gradient(y, tx, w):
    """
        Params :
            y : N * 1
            tx : N * D
            w : D * 1
        Returns :
            array of size D * 1

    """
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / y.shape[0]
    return grad

def loglikelihood_gradient(y, tx, w, lambda_):
    """compute the gradient of loss."""
    return tx.T.dot(sigmoid(tx.dot(w)) - y) + 2 * lambda_ * w