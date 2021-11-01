# -*- coding: utf-8 -*-
import numpy as np
from proj1_helpers import *
from costs import *
from gradient import *
from utils import *

def ridge_regression(y, tx, lambda_):
    """
    Calculates the weights for a ridge regression L2
    Params : 
        - y : array N * 1
        - tx : array N * D
        - lambda_ : scalar
    Returns :
        array of size D
    """
    w = np.linalg.solve(
            tx.T.dot(tx) + 2 * y.shape[0] * lambda_ * np.identity(tx.shape[1]), 
            tx.T.dot(y)
            )
    return w, compute_mse(y,tx,w)


def least_squares(y, tx):
    """
    Calculates the optimal weights by using least squares regression
    Params : 
        - y : array N * 1
        - tx : array N * D
    Returns :
        array of size D
    """
    return ridge_regression(y, tx, 0)


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
        Calculates the optimal weights by using least squares gradient descent
        Params : 
        - y : array N * 1
        - tx : array N * D
        - initial_w : array D * 1
        - max_iters : int
        - gamma : float
    Returns :
        array of size D
    """
    return least_squares_SGD(y, tx, initial_w, max_iters, gamma, batchsize = y.shape[0])

def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batchsize = 1):
    """
        Calculates the optimal weights by using least squares stochastic gradient descent
        Params : 
        - y : array N * 1
        - tx : array N * D
        - initial_w : array D * 1
        - max_iters : int
        - gamma : float
        - batchsize : int
    Returns :
        array of size D
    """
    w = initial_w.flatten()

    for _ in range(max_iters):
        batch = batch_iter(y, tx, batchsize)
        
        for minibatch_y, minibatch_tx in batch:
            grad = GD_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma * grad

    return w, compute_mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
        Calculates the optimal weights by using logistic regression gradient descent
        Params : 
        - y : array N * 1
        - tx : array N * D
        - initial_w : array D * 1
        - max_iters : int
        - gamma : float
    Returns :
        array of size D
    """
    return reg_logistic_regression(y, tx, 0, initial_w, max_iters, gamma)
    
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
        Calculates the optimal weights by using penalized logistic regression gradient descent
        Params : 
        - y : array N * 1
        - tx : array N * D
        - lambda_ : float
        - initial_w : array D * 1
        - max_iters : int
        - gamma : float
    Returns :
        array of size D
    """
    w = initial_w.flatten()
    y = y.flatten()
    for _ in range(max_iters):
        grad = loglikelihood_gradient(y, tx, w, lambda_)
        w = w - gamma * grad

    return w, compute_loglikelihood(y, tx, w, lambda_)

