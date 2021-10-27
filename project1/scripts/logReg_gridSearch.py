
import numpy as np

from preprocessing_helpers import *

from plots import visualization

from proj1_helpers import *

# Next strategy to try: use concurrent.futures (also in Python 3 standard library)
import concurrent.futures

import copy

import sys

import time

# @ TODO use most recent version of logistic regression
# @ TODO Argument for logistic reg: distribution from which w_initial should
# be sampled
# @ TODO store_losses should be false
# @ TODO store final loss
# @ TODO k_fold validation

#utils for ClassifierLogisticRegression

def sigmoid(t):
    """Compute the sigmoid function on t."""
    return 1./ (1+ np.exp(-t))


def der_sigmoid(t):
    """Compute derivtive of sigmoid on t"""
    return sigmoid(t)*(1-sigmoid(t))


def log_likelihood_loss(y, tx, w, lambda_):
    """compute the loss: negative log likelihood."""
    pred = sigmoid(tx.dot(w)) 
    return 1/2 * (((y - pred)**2).mean() + lambda_ * np.linalg.norm(w))


def calculate_gradient(y, x, w, lambda_):
    """compute the gradient of log_likelihood_loss wrt weights."""
    temp = y.ravel()
    arg = x@w
    s = sigmoid(arg)
    gradient = x.T@(s - temp)
    return gradient + lambda_ * w


def learning_by_gradient_descent(y, tx, w, gamma, lambda_, return_gradient = False):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    w = w.ravel()
    #compute loss
    loss = log_likelihood_loss(y, tx, w, lambda_)
    #copmute gradient
    grad = calculate_gradient(y, tx, w, lambda_)
    #do gradient step 
    w -= gamma*grad

    #if required, return also the gradient
    if return_gradient:
        return loss, w, grad
    return loss, w

#Logistic Regression
class ClassifierLogisticRegression():

    def update_params(self):
        """Update the dictionary containng the parameters"""
        self.params['name'] = self.name
        self.params['lambda_'] = self.lambda_
        self.params['regulairizer'] = self.regularizer
        self.params['gamma'] = self.gamma
        self.params['max_iterations'] = self.max_iterations
        self.params['threshold'] = self.threshold
    
    def __init__(self, lambda_, regularizer, gamma, max_iterations , threshold):
        """ 
            Sets parameters for logistic regression
            Argument:
                - gamma (float)
                - n_iterations (int)
        """

        #A dictionary containing the relevant parameters of the classifier 
        self.params = dict()

        #name of the classifier 
        self.name = 'LogisticRegression'
        
        #weight of the regularization term in the loss
        self.lambda_= lambda_
        
        #kind of regularizer: L1, L2 or None
        self.regularizer = regularizer

        #the step in gradient descent
        self.gamma = gamma

        #the maximum number of iterations
        self.max_iterations = max_iterations

        #threshold in gradient descent
        self.threshold = threshold
        
        self.update_params()

    def train(self, y_train, tx_train, batch_size = -1, verbose = True, tx_validation = None, y_validation = None, store_gradient = False):
        """ 
            Trains the model. It learns a new w with logistic regression. 
            Arguments:
                - tx_train: ndarray matrix of size N*D
                - y_train: ndarray matrix of size D*1
            Hypothesis: tx_train ndd y_train have the same length
        """
        #initialize the weights
        # self.w = np.zeros(tx_train.shape[1])

        # NEW: generate a random vector of values between -1 and 1 and use as w

        self.w = np.random.uniform(-1, 1, tx_train.shape[1]) 

        # also store a deep copy of that random vector in 'w_initial'

        w_initial = copy.deepcopy(self.w)

        #store the losses over 1 complete iteration (epoch)
        self.losses = []
        #store the prediction accuracies if validation tests are inputted:
        if (tx_validation is not None) and (y_validation is not None):
            self.pred_accuracies_train = []
            self.pred_accuracies_validation = []
        #store the norm of the gradient if required
        if store_gradient:
            self.stored_gradients = []

        #initiazlie the number of samples
        N = tx_train.shape[0]
        #initialize the batch size
        if batch_size == -1:
            batch_size = N

        #iterate over the dataset
        for iter in range(self.max_iterations):
            
            #loss accumulated over many batches
            acc_loss = 0
            for b in range(0, N, batch_size):  
                
                #perform a gradient step over a batch
                #if required, get also the gradient
                if store_gradient:
                    l, self.w, grad = learning_by_gradient_descent(
                        y_train[b:b+batch_size], 
                        tx_train[b:b+batch_size], 
                        self.w, 
                        self.gamma,
                        self.lambda_, 
                        return_gradient = True)
                    
                else:
                    l, self.w = learning_by_gradient_descent(
                        y_train[b:b+batch_size], 
                        tx_train[b:b+batch_size], 
                        self.w, 
                        self.gamma,
                        self.lambda_
                        )
            
                #update accumulated loss
                acc_loss += l

            #output the loss if verbose
            if verbose and iter % 100 == 0:
                print("Current iteration={a}, loss={b}".format(a=iter, b=acc_loss))
            
            #if required, store the predictions log
            if (tx_validation is not None) and (y_validation is not None):
                self.pred_accuracies_train += [(self.predict(tx_train) == y_train).mean()]
                self.pred_accuracies_validation += [(self.predict(tx_validation) == y_validation).mean()]

            #if required, store the norm of the gradient
            if store_gradient:
                self.stored_gradients += [np.linalg.norm(grad)]

            #store the loss over an iteration
            self.losses += [acc_loss]
            

            #check if convergence has been achieved
            if len(self.losses) > 1 and np.abs(self.losses[-1] - self.losses[-2]) < self.threshold:
            
                #update internal parameters and exit
                self.params['losses'] = self.losses
                self.params['weights'] = self.w
                print('hit thresh')

                #if accuracies were required:
                if (tx_validation is not None) and (y_validation is not None):
                    self.params['accuyracues_while_training_train'] = self.pred_accuracies_train
                    self.params['accuyracues_while_training_validation'] = self.pred_accuracies_validation                    

                #if required, store the norm of the gradient
                if store_gradient:
                    self.params['stored_gradients'] = self.stored_gradients
                
                break

        #end of training: update internal parameters and exit
        self.params['losses'] = self.losses
        self.params['weights'] = self.w

        #if accuracies were required:
        if (tx_validation is not None) and (y_validation is not None):
            self.params['accuyracues_while_training_train'] = self.pred_accuracies_train
            self.params['accuyracues_while_training_validation'] = self.pred_accuracies_validation
    
        #if required, store the norm of the gradient
        if store_gradient:
            self.params['stored_gradients'] = self.stored_gradients

        return w_initial
    
    def predict(self, x):
        """ 
            returns a list of predictions
            Argument:
                - x: a sample vector 1*D 
            Returns : 
                Array[int] 
        """
        pred = sigmoid(x.dot(self.w)) 
        pred = np.rint(pred)
        return pred

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def preprocessing(y, tx, nan_strategy, standardize_=True, outliers = False):
  """
  Do the preprocessing of the data.
  Argument:
      - y : of shape (N, )
      - tx: of shape (N, D)
      - nan_strategy: string. Defines how we handle the data. One of the following:
          1. 'NanToMean', replaces NaNs with the mean
          2. 'NanToMedian', replaces the NaNs with the median
          3. 'RemoveNan', removes the rows containing the NaNs
          4. 'RemoveNanFeatures' removes the columns with NaNs
          5. 'NanTo0', replaces the NaNs with 0
      - standardize: standardizes the data
      - outliers TODO

  Returns : 
      -(res_y, res_x) : tuple of processed data

  """
  NANVAL = -998
  
  #TODO : outliers
  res_x = tx
  res_y = y
  res_x = np.where(res_x < NANVAL, np.NaN, res_x)

  indices = np.where(np.isnan(res_x))
  if nan_strategy=='NanToMean':
    # Replace with mean
    means = np.nanmean(res_x, axis=0)
    res_x[indices] = np.take(means, indices[1]) 
  elif nan_strategy=='NanToMedian':
    # Replace with median
    medians = np.nanmedian(res_x, axis=0)
    res_x[indices] = np.take(medians, indices[1])
  elif nan_strategy=='RemoveNan':
    # Remove the NaN
    rows_with_nan = ~np.isnan(res_x).any(axis=1)
    res_y, res_x = res_y[rows_with_nan], res_x[rows_with_nan]
  elif nan_strategy=='RemoveNanFeatures':
    # Remove columns with NaN
    columns_with_nan = ~np.isnan(res_x).any(axis=0)
    res_x = res_x[:,columns_with_nan]
  elif nan_strategy == 'OnlyNanFeatures':
    columns_wo_nan = np.isnan(res_x).any(axis=0)
    res_x = res_x[:,columns_wo_nan]
    rows_with_nan = ~np.isnan(res_x).any(axis=1)
    res_y, res_x = res_y[rows_with_nan], res_x[rows_with_nan]
  elif nan_strategy== 'NanTo0':
    # Replace with 0
    res_x = np.nan_to_num(res_x)
  else:
    raise Error('specify a correct strategy')


  if outliers : 
    #TODO remove outliers
    pass
  if standardize_: 
    res_x, _, _ = standardize(res_x)
  return res_y, res_x

def split_data(x, y, ratio, verbose = False):
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

    #real stuff now

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read in the data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TRAIN = '../data/train.csv' # due to directory structure, the data directory is now one directory above this one
TEST = '../data/test.csv'
y, x, ids_train = load_csv_data('../data/train.csv')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For exploring relationship of parameters in their effect on accuracy, set up lists of possible values for these parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Create a list of tuples, with each tuple representing a random combination of parameters

triplets = []

# Number of features: 
# * 30 (original dataset), 
# * 90 (original dataset + polynomials of deg 2), 
# * 120 (original dataset + polynomials of deg 2 + polynomials of deg 3)
x_deg2 = build_poly_standard(x, 2)

x_deg3 = build_poly_standard(x, 3)

datasets = [x, x_deg2, x_deg3]

lambdas = np.logspace(-4, 0, 40)

# generate an arbitrary amount of models

for i in range(40):

    l_rate = np.random.uniform(10**(-7), 10**(-1))

    # l_rate = np.random.uniform(0.075,0.085) # better use a logarithmic scale?

    b_size = np.random.randint(1000, 100001)

    dataset_ind = np.random.randint(0,3)

    triplets.append((l_rate, dataset_ind, b_size, lambdas[i]))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Train a model for a specific combination of l_rate, dataset, and b_size, and return the rediction accuracy 

def train_model(l_rate, dataset, b_size, lambda__):

    # @ TODO perform k_fold cross-validation here
    
    y_train, tx_train = preprocessing(y, datasets[dataset], 'NanToMean', standardize_=True) 
    
    clf = ClassifierLogisticRegression(lambda_ = lambda__, regularizer = None, gamma= l_rate, max_iterations = 1000, threshold = 0)
    
    y_train = (y_train+1)/2
    
    print('Sanity check :n = ', tx_train.shape[0], 'D = ', tx_train.shape[1])
    
    tx_train, y_train, tx_validation, y_validation = split_data(tx_train, y_train, ratio = 0.7, verbose = False)
    
    w_initial = clf.train(y_train, tx_train, verbose = True, batch_size = b_size, tx_validation = tx_validation, y_validation = y_validation, store_gradient=False)

    # @ TODO return train + test accuracy + losses incl. SD

    return (clf.predict(tx_train) == y_train).mean(), w_initial

# Worker function, i.e. the function that will be performed by each process

def worker(triplet):

    l_rate, dataset_ind, b_size, lambda_ = triplet

    # @TODO add w_initial to clf.params

    accuracy, w_initial = train_model(l_rate, dataset_ind, b_size, lambda_)
    print(f'Learning rate: {l_rate}, dataset-nr: {dataset_ind}, batch-size: {b_size}, lambda: {lambda_}, accuracy: {accuracy}')

    # NEW: also store w_initial, as it is now randomly generated in 'train'-method in log reg classifier
    
    return ({"dataset": dataset_ind, "batch_size": b_size, "learning rate": l_rate, "lambda": {lambda_}, "w_initial": w_initial, "accuracy": accuracy})

if __name__ == '__main__':

    # using this library now, makes parallelization embarassingly easy

    t0 = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:

        result = executor.map(worker, triplets)

        res_list = list(result)

        print(res_list)

        # print(sys.getsizeof(tuple(result)))

        print(len(res_list))

        # @TODO 'result' is currently a tuple of dictionaries, should be transformed to np array for plotting purposes

    t1 = time.time()

    total = t1-t0

    print(total)
