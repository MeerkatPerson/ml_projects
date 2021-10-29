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
        expanded_noninteractive_deg2 = np.append(
            x, new_features_noninteractive_deg2, axis=1
        )

        # Add to the list of datasets
        datasets.append(expanded_noninteractive_deg2)
        #print(datasets[1])

        # Currently picking only 8 random batches of interactive terms of degree 2 because there's ~45 of them ...
        k = 2

        for i in range(0, 0):

            j = np.random.randint(2, len(k_exponents_deg2))

            # Generate a batch of interactive terms of degree two
            new_feature_batch_interactive_deg2 = gen_new_features(
                x, k_exponents_deg2[j]
            )

            # Augment the dataset containing non-interactive terms up to degree 2 with a batch of 10 interactive terms of degree two
            expanded_interactive_deg2 = np.append(
                expanded_noninteractive_deg2, new_feature_batch_interactive_deg2, axis=1
            )

            # append
            datasets.append(expanded_interactive_deg2)
            print(datasets[k])

            k += 1

            print("Added a new dataset with 10 interactive terms of degree 2 !")

        # TODO pickle list of datasets so that we are able to relate the dataset indices output by the model to the actual dataset
        #     it was generated with

        # grab the dataset indices
        dataset_indices = np.arange(0, len(datasets), 1).tolist()

        # generate learning rates
        l_rates = np.logspace(-5, -1, 10)
        l_rates = np.array([1e-6])

        # generate batch sizes on a linear scale
        # currently not using because of k-fold crossvalidation

        b_sizes = np.linspace(start=1000, stop=x.shape[0], num=10, dtype=int)
        b_sizes = np.array([50000])

        # list of possiblities for sampling initial values of w

        initial_w_distributions = ["uniform", "normal", "log", "zero"]
        initial_w_distributions = ['uniform']

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

        # (1) Pre-process
        y_train, tx_train = preprocess(
            self.y, self.datasets[dataset_idx], "NanToMean", standardize_=True
        )

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
            y_train_k = (y_train_k + 1) / 2
            y_test_k = y_train[test]
            y_test_k = (y_test_k + 1) / 2
            x_train_k = tx_train[train]
            x_test_k = tx_train[test]

            # (5) Compute clf
            clf = ClassifierLogisticRegression(
                lambda_=0,
                regularizer=None,
                gamma=l_rate,
                max_iterations=1000,
                w_sampling_distr=initial_w_dist,
                threshold=0,
            )

            # Batch size now -1, which results in batch_size = N, as using k-fold
            clf.train(
                y_train=y_train_k,
                tx_train=x_train_k,
                batch_size=b_size,
                verbose=True,
                tx_validation=x_test_k,
                y_validation=y_test_k,
                store_gradient=False,
            )

            acc_tr.append((clf.predict(x_train_k) == y_train_k).mean())
            acc_te.append((clf.predict(x_test_k) == y_test_k).mean())
            losses_tr.append(mse_loss(y_train_k, x_train_k, clf.w, clf.lambda_))
            losses_te.append(mse_loss(y_test_k, x_test_k, clf.w, clf.lambda_))

        return (
            sum(acc_tr) / (len(acc_tr)),
            sum(acc_te) / (len(acc_te)),
            sum(losses_tr) / (len(losses_tr)),
            sum(losses_te) / (len(losses_te)),
        )

    # Worker function, i.e. the function that will be performed by each process
    def worker(self, tuple_):

        dataset, l_rate, b_size, initial_w_dist = tuple_

        print("Worker starting to compute model!")

        acc_tr, acc_te, losses_tr, losses_te = self.train_model(
            dataset, l_rate, b_size, initial_w_dist
        )
        print(
            f"dataset: {dataset}, learning rate: {l_rate}, batch_size: {b_size}, w_initial distribution: {initial_w_dist}, acc_tr: {acc_tr}, acc_te: {acc_te}, losses_tr: {losses_tr}, losses_te: {losses_te}"
        )

        return {
            "dataset": dataset,
            "learning rate": l_rate,
            "batch_size": b_size,
            "w_initial distr": initial_w_dist,
            "acc_tr": acc_tr,
            "acc_te": acc_te,
            "losses_tr": losses_tr,
            "losses_te": losses_te,
        }


if __name__ == "__main__":

    gss = GridSearch_Simulation("../data/train.csv", "../data/test.csv")

    t0 = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:

        result = executor.map(gss.worker, gss.param_tuples)

        res_list = list(result)

        print(res_list)

        # print(sys.getsizeof(tuple(result)))

        print(len(res_list))

        # @TODO 'result' is currently a tuple of dictionaries, should be transformed to np array for plotting purposes

    t1 = time.time()

    total = t1 - t0

    print(total)
