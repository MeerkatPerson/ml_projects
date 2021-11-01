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
    loss = mse_loss(y, tx, w, lambda_)
    #loss = loglikelihood_loss(y, tx, w, lambda_)
    
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




