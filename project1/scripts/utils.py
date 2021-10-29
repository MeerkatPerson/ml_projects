import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This module contains utils used by the classifiers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# mostly ClassifierCentroids-related functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def build_centroids(y, x):
    """
    Given a dataset x with labels y, 
    retiurns a list containing a tuples of the kind:
    (class label, centroid of the class)
    """
    res = []
    for cl in set(y):
      res.append((cl, np.mean(x[y==cl], axis = 0)))
    return res

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# mostly ClassifierLogisticRegression-related functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def sigmoid(t):
    """Compute the sigmoid function on t."""
    return 1./ (1+ np.exp(-t))


def der_sigmoid(t):
    """Compute derivtive of sigmoid on t"""
    return sigmoid(t)*(1-sigmoid(t))


def mse_loss(y, tx, w, lambda_):
    """compute the loss: negative log likelihood."""
    pred = sigmoid(tx.dot(w)) 
    return ((y - pred)**2).mean() + lambda_ * (w**2).mean()

def calculate_gradient(y, x, w, lambda_):
    """compute the gradient of log_likelihood_loss wrt weights."""
    temp = y.ravel()
    arg = x@w
    s = sigmoid(arg)
    gradient = x.T@(s - temp)
    return gradient + 2 * lambda_ * w


def learning_by_gradient_descent(y, tx, w, gamma, lambda_, return_gradient = False):
    """
    Performs one step of gradient descent using logistic regression.
    Arguments:
        - y: output of the model
        - tx: input of the model
        - gamma: learning rate
        - return_gradient: Boolean. If true returns the gradient computed
    Return:
        - loss: the loss of the model, comuted using y, tx and the updated weights
        - w: the updated weights
        - grad: only if return_gradient == True. Returns the computed gradient
    Notice: the loss returned (mse) is not the one used to compute the gradient (loglikelihood).
    TODO: add the possibility to return a gradient usng regularizers
    """
    w = w.ravel()
    grad = calculate_gradient(y, tx, w, lambda_)
    w -= gamma*grad
    loss = mse_loss(y, tx, w, lambda_)

    #if required, return also the gradient
    if return_gradient:
        return loss, w, grad
    return loss, w

# Build k indices for k-fold cross-validation

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)