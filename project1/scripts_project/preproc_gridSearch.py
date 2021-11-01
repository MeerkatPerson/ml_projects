
import numpy as np

from plots import visualization

from proj1_helpers import *

# Next strategy to try: use concurrent.futures (also in Python 3 standard library)
import concurrent.futures

import time

from preprocessing import *

from utils import *

from classifiers import *

from validation import *

from features_ext import *

from itertools import product

import pickle

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read in the data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Yup I too have gone object-oriented ... 

class PreProc_GridSearch:

    def __init__(self, train, test):

        self.train = train
        self.test = test

        self.y, self.x, self.param_tuples = self.get_data_and_params()

    def get_data_and_params(self):
    
        y, x, ids_train = load_csv_data(self.train)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # For exploring relationship of parameters in their effect on accuracy, set up lists of possible values for these parameters
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Create a list of tuples, each corresponding to one preprocessing strategy

        nan_treatment = ['NanToMean','NanToMedian','RemoveNan','OnlyNanFeatures','NanTo0']

        # additional_nan_feature = [True,False]

        standardize = [True,False]

        tuples = list(product(nan_treatment, standardize))

        # print(tuples)

        # Generate non-interactive features of deg 2 and append

        k_exponents_deg2 = exponents(x.shape[1], 2, 10, non_interaction_first=True)

        # Now generate the non-interactive terms of degree two
        new_features_noninteractive_deg2 = gen_new_features(x, k_exponents_deg2[0])

        # Augment the original dataset with the non-interactive terms of degree two
        expanded_noninteractive_deg2 = np.append(x, new_features_noninteractive_deg2, axis = 1 )

        return y, expanded_noninteractive_deg2, tuples

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Train a model for a specific combination of l_rate, dataset, and b_size, and return the rediction accuracy 

    def train_model(self, nan_treatment, standardize):

        tx_train = copy.deepcopy(self.x)

        # (1) Pre-process

        y_train, tx_train = preprocess(self.y, tx_train, nan_treatment, standardize_=standardize) 
        
        print('Sanity check :n = ', tx_train.shape[0], 'D = ', tx_train.shape[1])

        # (2) Build k indices

        k_fold = 4 # using a default from the lab, can of course be tweaked

        seed = 1   # using a default from the lab, can of course be tweaked 

        k_indices = build_k_indices(y_train, k_fold, seed)

        print(f'k-fold dimensions: {k_indices.shape[0]}, {k_indices.shape[1]}')

        # (3) Set up arrays in which the results of the k-fold will be stored

        acc_tr_logreg = []
        acc_te_logreg = []

        losses_tr_logreg = []
        losses_te_logreg = []

        for k in range(k_fold):

            test = k_indices[k]

            # print(f'k-test dimensions: {test.shape[0]}, {test.shape[1]}')
    
            train = []
    
            for i in range(len(k_indices)):
                
                if i is not k:
                
                    train.extend(k_indices[i])

            # (4) Get the train and test sets according to current k_index
    
            y_train_k = y_train[train]

            y_train_k = (y_train_k+1)/2

            y_test_k = y_train[test]

            y_test_k = (y_test_k+1)/2
    
            x_train_k = tx_train[train]

            x_test_k = tx_train[test]

            # (5) Compute clf

            clf = ClassifierLogisticRegression(lambda_ = 0, regularizer = None, gamma= 0.01, max_iterations = 1000, min_max_iterations = 100,w_sampling_distr = 'normal', threshold = 0)

            # Batch size now -1, which results in batch_size = N, as using k-fold

            clf.train(y_train = y_train_k, tx_train = x_train_k, batch_size = 50000, verbose = True, tx_validation = x_test_k, y_validation = y_test_k, store_gradient=False)

            # Append logreg results

            acc_tr_logreg.append( (clf.predict(x_train_k) == y_train_k).mean() )

            acc_te_logreg.append( (clf.predict(x_test_k) == y_test_k).mean() )

            losses_tr_logreg.append(mse_loss(y_train_k, x_train_k, clf.w, clf.lambda_))

            losses_te_logreg.append(mse_loss(y_test_k, x_test_k, clf.w, clf.lambda_))

        return sum(acc_tr_logreg)/(len(acc_tr_logreg)), sum(acc_te_logreg)/(len(acc_te_logreg)), sum(losses_tr_logreg)/(len(losses_tr_logreg)), sum(losses_te_logreg)/(len(losses_te_logreg))

    # Worker function, i.e. the function that will be performed by each process

    def worker(self, tuple_):

        nan_treatment, standardize = tuple_

        try: 

            print("Worker starting to compute model!")

            acc_tr_logreg, acc_te_logreg, losses_tr_logreg, losses_te_logreg  = self.train_model(nan_treatment, standardize)
            
            print(f'LogReg: nan_treatment: {nan_treatment}, standardize: {standardize}, acc_tr: {acc_tr_logreg}, acc_te: {acc_te_logreg}, losses_tr: {losses_tr_logreg}, losses_te: {losses_te_logreg}')
            
            return ({"nan_treatment": nan_treatment, "standardize": standardize, "acc_tr_logreg": acc_tr_logreg, "acc_te_logreg": acc_te_logreg, "losses_tr_logreg": losses_tr_logreg, "losses_te_logreg": losses_te_logreg})
            
        except: 

            print("Error computing model!")

            print(f'Error occurred with: nan_treatment: {nan_treatment}, standardize: {standardize}')
            return ({"nan_treatment": nan_treatment, "standardize": standardize, "acc_tr": 'unknown', "acc_te": 'unknown', "losses_tr": 'unknown', "losses_te": 'unknown'})

if __name__ == '__main__':

    # Create results directory if it doesn't already exist
    import os

    if not os.path.exists('../results'):
        os.makedirs('../results')

    gss = PreProc_GridSearch('../data/train.csv', '../data/test.csv') 

    t0 = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:

        result = executor.map(gss.worker, gss.param_tuples)

        res_list = list(result)

        print(res_list)

        print(len(res_list))

        with open('../results/preproc_gridsearch_res.txt', 'wb') as fh:
            pickle.dump(res_list, fh)

    t1 = time.time()

    total = t1-t0

    print(total)
