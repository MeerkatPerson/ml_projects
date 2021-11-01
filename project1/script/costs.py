# -*- coding: utf-8 -*-
import numpy as np


# TESTED
def sigmoid(t):
    """Compute the sigmoid function on t."""
    return 1./ (1.+ np.exp(-t))

def compute_loglikelihood(y, tx, w, lambda_):

    '''compute the loss: negative log likelihood.'''
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    loss = - loss
    return loss + lambda_ * w.T.dot(w)

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    err = y - tx.dot(w)
    return 0.5 * err.dot(err) / y.shape[0]

