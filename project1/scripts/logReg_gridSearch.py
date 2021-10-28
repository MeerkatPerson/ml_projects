
import numpy as np

# from preprocessing_helpers import *

from preprocess import *

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
    
        y, x, ids_train = load_csv_data(self.train)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # For exploring relationship of parameters in their effect on accuracy, set up lists of possible values for these parameters
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Create a list of tuples, with each tuple representing a random combination of parameters

        # Generate exponents of degree 2 for now
        # k_exponents_deg2[0] will contain the non-interactive terms of degree 2;
        # k_exponents_deg2[1] ... k_exponents_deg2[len(k_exponents_deg2)-1] will contain random sets of 10 interactive terms of degree 2
        # arguments of the 'exponents'-function (which resides in features_ext.py): 
        # * number of original features (= # columns of the original data matrix)
        # * degree
        # * number of terms in each of the sets of interactive terms in k_exponents_deg2[1] ... k_exponents_deg2[len(k_exponents_deg2)-1]
        # * non_interaction_first=True: this means that k_exponents_deg2[0] will contain the non-interactive terms of degree 2.

        k_exponents_deg2 = exponents(x.shape[1], 2, 10, non_interaction_first=True)

        # First add just the 'normal' unaugmented data matrix to the list of datasets.
        datasets = [x]

        # Now generate the non-interactive terms of degree two
        new_features_noninteractive_deg2 = gen_new_features(x, k_exponents_deg2[0])

        # Augment the original dataset with the non-interactive terms of degree two
        expanded_noninteractive_deg2 = np.append(x, new_features_noninteractive_deg2, axis = 1 )

        # Add to the list of datasets

        datasets.append(expanded_noninteractive_deg2)

        # Currently picking only 8 random batches of interactive terms of degree 2 because there's ~45 of them ...

        for i in range(0,8):

            j = np.random.randint(2, len(k_exponents_deg2))

            # Generate a batch of interactive terms of degree two
            new_feature_batch_interactive_deg2 = gen_new_features(x, k_exponents_deg2[j])

            # Augment the original dataset with the non-interactive terms of degree two
            expanded_interactive_deg2 = np.append(expanded_noninteractive_deg2, new_feature_batch_interactive_deg2, axis = 1 )

            datasets.append(expanded_interactive_deg2)

            print("Added a new dataset with 10 interactive terms of degree 2 !")

        #TODO pickle list of datasets so that we are able to relate the dataset indices output by the model to the actual dataset
        #     it was generated with

        # grab the dataset indices

        dataset_indices = np.arange(0,len(datasets),1).tolist()

        # generate learning rates

        l_rates = np.logspace(-5,-1,10)

        # generate batch sizes on a linear scale

        b_sizes = np.linspace(start=1000, stop=x.shape[0], num=10, dtype=int)

        # list of possiblities for sampling initial values of w

        initial_w_distributions = ["uniform","normal","log","zero"]

        # Get the crossproduct of all these parameters

        tuples = list(product(dataset_indices, l_rates, b_sizes, initial_w_distributions))

        print(tuples)

        return y, datasets, tuples

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Train a model for a specific combination of l_rate, dataset, and b_size, and return the rediction accuracy 

    def train_model(self, dataset_idx, l_rate, b_size, initial_w_dist):

        # @ TODO perform k_fold cross-validation here
        
        # print(dataset_idx)

        # print(len(datasets))

        y_train, tx_train = preprocess(self.y, self.datasets[dataset_idx], 'NanToMean', standardize_=True) 
        
        clf = ClassifierLogisticRegression(lambda_ = 0, regularizer = None, gamma= l_rate, max_iterations = 1000, w_sampling_distr = initial_w_dist, threshold = 0)
        
        y_train = (y_train+1)/2
        
        print('Sanity check :n = ', tx_train.shape[0], 'D = ', tx_train.shape[1])
        
        tx_train, y_train, tx_validation, y_validation = split_data(tx_train, y_train, ratio = 0.7, verbose = False)
        
        clf.train(y_train, tx_train, verbose = True, batch_size = b_size, tx_validation = tx_validation, y_validation = y_validation, store_gradient=False)

        # @ TODO return train + test accuracy + losses incl. SD

        return (clf.predict(tx_train) == y_train).mean()

    # Worker function, i.e. the function that will be performed by each process

    def worker(self, tuple_):

        dataset, l_rate, b_size, initial_w_dist = tuple_

        print("Worker starting to compute model!")

        # @TODO add w_initial to clf.params

        accuracy = self.train_model(dataset, l_rate, b_size, initial_w_dist)
        print(f'dataset: {dataset}, learning rate: {l_rate}, batch-size: {b_size}, w_initial distribution: {initial_w_dist}, accuracy: {accuracy}')

        # NEW: also store w_initial, as it is now randomly generated in 'train'-method in log reg classifier
        
        return ({"dataset": dataset, "batch_size": b_size, "learning rate": l_rate, "w_initial distr": initial_w_dist, "accuracy": accuracy})

if __name__ == '__main__':

    gss = GridSearch_Simulation('../data/train.csv', '../data/test.csv') 

    t0 = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:

        result = executor.map(gss.worker, gss.param_tuples)

        res_list = list(result)

        print(res_list)

        # print(sys.getsizeof(tuple(result)))

        print(len(res_list))

        # @TODO 'result' is currently a tuple of dictionaries, should be transformed to np array for plotting purposes

    t1 = time.time()

    total = t1-t0

    print(total)
