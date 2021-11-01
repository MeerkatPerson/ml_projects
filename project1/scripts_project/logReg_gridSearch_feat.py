import numpy as np

# from preprocessing_helpers import *

from preprocessing import *

from plots import visualization

from proj1_helpers import *

# Next strategy to try: use concurrent.futures (also in Python 3 standard library)
import concurrent.futures

import time
from utils import *
from classifiers import *
from validation import *
from features_ext import *
from itertools import product

import pickle

import zipfile

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read in the data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Yup I too have gone object-oriented ...
class GridSearch_Simulation:
    def __init__(self, train, test):

        self.train = train
        self.test = test

        self.y, self.datasets, self.param_tuples = self.get_data_and_params()

    def get_data_and_params(self):

        #import and preprocess the data
        y, x, ids_train = load_csv_data(self.train)
        y, x = preprocess(
            y, x, "NanToMean", standardize_=False
        )
        y = (y + 1)/2.

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # For exploring relationship of parameters in their effect on accuracy, set up lists of possible values for these parameters
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Create a list of tuples, with each tuple representing a random combination of parameters

        # Generate exponents of degree 2 for now
        #BELOW DESCRIPTION OUTDATED
        # k_exponents_deg2[0] will contain the non-interactive terms of degree 2;
        # k_exponents_deg2[1] ... k_exponents_deg2[len(k_exponents_deg2)-1] will contain random sets of 10 interactive terms of degree 2
        # arguments of the 'exponents'-function (which resides in features_ext.py):
        # * number of original features (= # columns of the original data matrix)
        # * degree
        # * number of terms in each of the sets of interactive terms in k_exponents_deg2[1] ... k_exponents_deg2[len(k_exponents_deg2)-1]
        # * non_interaction_first=True: this means that k_exponents_deg2[0] will contain the non-interactive terms of degree 2.

        

        # First add just the 'normal' unaugmented data matrix to the list of datasets.
        datasets = []

        #build the expanded polynomial to degree 2
        x_2, _ = build_poly_and_standardize(x, x, 2)
        datasets += [x_2]

        #build the expanded polynomial to degree 3
        x_3, _ = build_poly_and_standardize(x, x, 3)
        datasets += [x_3]
               
        #add a subset of non-interacting terms
        k_exponents_deg2 = exponents(x.shape[1], 2, 10, non_interaction_first=True)
        #Fabrizio: 1:20
        #Theresa: 20:40
        for i in range(5,10):

            # Generate a batch of interactive terms of degree two
            new_feature = gen_new_features(x, k_exponents_deg2[i])
            new_feature, _, _ = standardize(new_feature)
            
            # Augment the dataset containing non-interactive terms up to degree 2 with a batch of 10 interactive terms of degree two
            expanded_w_interactions = np.append(
                x_3, new_feature, axis=1
            )

            # append
            datasets.append(expanded_w_interactions)
            print("Added a new dataset with 10 interactive terms of degree 2 !")

        # pickle list of datasets so that we are able to relate the dataset indices output by the model to the actual dataset
        #     it was generated with

        with open('../results/dataset_list.txt', 'wb') as fh:
            pickle.dump(datasets, fh)


        # grab the dataset indices
        dataset_indices = np.arange(0,len(datasets),1).tolist()

        #####################################################################
        # Choose the learning rates
        #####################################################################
        l_rates = np.array([6.8e-5])
    

        ####################################################################
        # Choose the batch sizes
        ##################################################################
        b_sizes = np.array([20000])

        
        ################################################################
        # Choose initializations
        ###################################################################
        initial_w_distributions = ["normal"]

        # Get the crossproduct of all these parameters
        tuples = list(
            product(dataset_indices, l_rates, b_sizes, initial_w_distributions)
        )

        print('number of datasets:', len(datasets))
        print('combinations of hyperparameters', len(tuples))

        return y, datasets, tuples

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Train a model for a specific combination of l_rate, dataset, and b_size, and return the rediction accuracy

    def train_model(self, dataset_idx, l_rate, b_size, initial_w_dist):

        # (1) Get values
        y_train = self.y
        tx_train = self.datasets[dataset_idx]

        print("Sanity check : n_variables = ", tx_train.shape[0], "n_features = ", tx_train.shape[1])

        # (2) Build k indices
        k_fold = 4  # using a default from the lab, can of course be tweaked
        seed = 1  # using a default from the lab, can of course be tweaked
        k_indices = build_k_indices(y_train, k_fold, seed)
        print(f"k-fold dimensions: {k_indices.shape[0]}, {k_indices.shape[1]}")

        # (3) Set up arrays in which the results of the k-fold will be stored
        acc_tr = []
        acc_te = []

        losses_tr = []
        losses_te = []

        for k in range(k_fold):
            test = k_indices[k]
            # print(f'k-test dimensions: {test.shape[0]}, {test.shape[1]}')

            train = []
            for i in range(len(k_indices)):
                if i is not k:
                    train.extend(k_indices[i])

            # (4) Get the train and test sets according to current k_index

            y_train_k = y_train[train]
            y_test_k = y_train[test]
            x_train_k = tx_train[train]
            x_test_k = tx_train[test]

            # (5) Compute clf
            clf = ClassifierLogisticRegression(
                lambda_=0,
                regularizer=None,
                gamma=l_rate,
                max_iterations=1500,
                min_max_iterations = 300,
                w_sampling_distr=initial_w_dist,
                threshold=1e-6,
            )

            # Batch size now -1, which results in batch_size = N, as using k-fold
            clf.train(
                y_train=y_train_k,
                tx_train=x_train_k,
                batch_size=b_size,
                verbose=True,
                store_losses=True,
                store_gradient=False,
            )

            acc_tr.append((clf.predict(x_train_k) == y_train_k).mean())
            acc_te.append((clf.predict(x_test_k) == y_test_k).mean())
            losses_tr.append(clf.params['losses'])
            losses_te.append(mse_loss(y_test_k, x_test_k, clf.w, clf.lambda_))

        return (
            acc_tr,
            acc_te,
            losses_tr,
            losses_te,
        )

    # Worker function, i.e. the function that will be performed by each process
    def worker(self, tuple_):

        #the tuple over which we make the grid search
        dataset, l_rate, b_size, initial_w_dist = tuple_

        try:

            print("Worker starting to compute model!")
            acc_tr, acc_te, losses_tr, losses_te  = self.train_model(dataset, l_rate, b_size, initial_w_dist)
            print(f'dataset: {dataset}, learning rate: {l_rate}, batch_size: {b_size}, w_initial distribution: {initial_w_dist}, acc_tr: {acc_tr}, acc_te: {acc_te}, losses_tr: {losses_tr}, losses_te: {losses_te}')      
            return ({"dataset": dataset, "learning rate": l_rate, "batch_size": b_size, "w_initial distr": initial_w_dist, "acc_tr": acc_tr, "acc_te": acc_te, "losses_tr": losses_tr, "losses_te": losses_te})

        except:

            print("Error computing model!")
            print(f'Error occurred with: dataset: {dataset}, learning rate: {l_rate}, batch_size: {b_size}, w_initial distribution: {initial_w_dist}')
            return ({"dataset": dataset, "learning rate": l_rate, "batch_size": b_size, "w_initial distr": initial_w_dist, "acc_tr": "unknown", "acc_te": "unknown", "losses_tr": "unknown", "losses_te": "unknown"})


if __name__ == '__main__':

    # Create results directory if it doesn't already exist
    import os

    if not os.path.exists('../results'):
        os.makedirs('../results')

    #initialize the object
    gss = GridSearch_Simulation('../data/train.csv', '../data/test.csv') 

    print('Grid search n parameters: ', len(gss.param_tuples))

    t0 = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:

        result = executor.map(gss.worker, gss.param_tuples)
        res_list = list(result)
        print(res_list)
        print(len(res_list))
        with open('../results/logreg_gridsearch_feat_2.txt', 'wb') as fh:
            pickle.dump(res_list, fh)

    t1 = time.time()
    total = t1-t0
    print('Time to run the entire grid search', total)

