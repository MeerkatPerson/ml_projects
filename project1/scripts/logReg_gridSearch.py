
import numpy as np

# from preprocessing_helpers import *

from preprocess import *

from plots import visualization

from proj1_helpers import *

# Next strategy to try: use concurrent.futures (also in Python 3 standard library)
import concurrent.futures

import copy

import sys

import time

from utils import *

from classifiers import *

from validation import *

from features_ext import *

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

triplets = [(1e-5,0,-1,0.1)]

# Number of features: 
# * 30 (original dataset), 
# * 90 (original dataset + polynomials of deg 2), 
# * 120 (original dataset + polynomials of deg 2 + polynomials of deg 3)

x_deg2 = build_poly_standard(x, 2)

x_deg3 = build_poly_standard(x, 3)

datasets = [x, x_deg2, x_deg3]

lambdas = np.logspace(-4, 0, 40)

# generate an arbitrary amount of models
'''
for i in range(40):

    l_rate = np.random.uniform(10**(-7), 10**(-1))

    # l_rate = np.random.uniform(0.075,0.085) # better use a logarithmic scale?

    b_size = np.random.randint(1000, 100001)

    dataset_ind = np.random.randint(0,3)

    triplets.append((l_rate, dataset_ind, b_size, lambdas[i]))
'''

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Train a model for a specific combination of l_rate, dataset, and b_size, and return the rediction accuracy 

def train_model(l_rate, dataset, b_size, lambda__):

    # @ TODO perform k_fold cross-validation here
    
    y_train, tx_train = preprocess(y, datasets[dataset], 'NanToMean', standardize_=True) 
    
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
