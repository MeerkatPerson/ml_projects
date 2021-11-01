import numpy as np
from numpy.lib import utils

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


def loglikelihood_loss(y, tx, w, lambda_):

    '''compute the loss: negative log likelihood.'''
    epsilon = 1e-10

    temp = y.ravel()
    arg = (tx@w).ravel()
    arg = sigmoid(arg)
    #print('Sigmoid is ufunc?', type(sigmoid))
    return -((temp*np.log(epsilon + arg)) + (1.-temp)*np.log( epsilon + 1. - arg)).sum()

""""
def loglikelihood_loss(y, tx, w, lambda_):
    '''Computes the loss using logistic loss function.

    Parameters
    ----------
    y: numpy array with shape (N,)
    The vector containing the correct predictions for samples
    tx: numpy array of numpy arrays with shape (N, D)
    The feature matrix of features
    w: numpy array of shape (D,)
    The current weight vector

    Returns
    -------
    mse: float
    The mean squared error
    '''
    epsilon = 1e-10

    y_pred = np.dot(tx, w)
    return np.sum(np.log(1. + np.exp(y_pred) + epsilon)) - np.dot(y.T, y_pred)
"""


def calculate_gradient(y, x, w, lambda_, normalize_gradient = False):
    """compute the gradient of log_likelihood_loss wrt weights."""
    temp = y.ravel()
    arg = x@w
    s = sigmoid(arg)
    #print('type s', s.dtype)
    gradient = x.T@(s - temp)
    if normalize_gradient:
        gradient = gradient/x.shape[0]
    return gradient + 2 * lambda_ * w


def learning_by_gradient_descent(y, tx, w, gamma, lambda_, return_gradient = False, normalize_gradient = False):
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
    grad = calculate_gradient(y, tx, w, lambda_, normalize_gradient)
    #print('dtype grad', grad.dtype, 'dtype gamma', type(gamma))
    w -= gamma*grad
    #print(w)
    #loss = mse_loss(y, tx, w, lambda_)
    loss = loglikelihood_loss(y, tx, w, lambda_)
    
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

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
def split_data(x, y, ratio, verbose = False):
    """
    Function to split the data in training and validation set.
    Shuffling of the original dataset is executed
    Arguments:
        -x: independent variable of the dataset
        -y: dependent variable of the dataset
        -ratio: the size ratio training/validation
        -verbose: prints the number of samples per subset
    Returns:
        - (sub_x_1, sub_y_1, sub_x_2, sub_y_2): tuple of subsets
    """
    n_samples = y.shape[0]

    indices = np.random.permutation(n_samples)
    sub_x_1 = x[indices][:int(ratio*n_samples)]
    sub_x_2 = x[indices][int(ratio*n_samples):]

    sub_y_1 = y[indices][:int(ratio*n_samples)]
    sub_y_2 = y[indices][int(ratio*n_samples):]

    if verbose:
        print('ration:\t', ratio)
        print('ratio of samples 1st subset:\t', np.round_((sub_y_1 == 1).sum()/(ratio*n_samples), decimals=2))
        print('ratio of samples 2nd subset:\t', np.round_((sub_y_2 == 1).sum()/((1-ratio)*n_samples), decimals=2))
    return sub_x_1, sub_y_1, sub_x_2, sub_y_2