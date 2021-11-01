from classifiers import *
from helpers import*
from proj1_helpers import load_csv_data
from preprocessing import *
from validation import *
from plots import *
from features_ext import *
import pickle

# import the data and preprocess
y, x, ids_train = load_csv_data(data_path = './train.csv')
y, x = preprocess(
    y, x, "NanToMean", standardize_=False
)
y = (y + 1)/2.

# (2) Build k indices
def cross_validate(tx_train, y_train):
        #define the parameters for training
        l_rate = 1e-2
        initial_w_dist = 'normal'
        b_size = 10000


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
            losses_tr.append(clf.params['losses'][-1])
            losses_te.append(mse_loss(y_test_k, x_test_k, clf.w, clf.lambda_))

        return (
            acc_tr,
            acc_te,
            losses_tr,
            losses_te,
        )

results = []
for d in range(2, 5):
        x_train, _ = build_poly_and_standardize(x, x, d)
        res = cross_validate(x_train, y)
        results += [res]

with open('overfit_degre_1.pkl', 'wb') as outp:
        pickle.dump(results, outp, pickle.HIGHEST_PROTOCOL)

results = []
for d in range(5, 8):
        x_train, _ = build_poly_and_standardize(x, x, d)
        res = cross_validate(x_train, y)
        results += [res]

with open('overfit_degre_2.pkl', 'wb') as outp:
        pickle.dump(results, outp, pickle.HIGHEST_PROTOCOL)

results = []
for d in range(8, 11):
        x_train, _ = build_poly_and_standardize(x, x, d)
        res = cross_validate(x_train, y)
        results += [res]

with open('overfit_degre_3.pkl', 'wb') as outp:
        pickle.dump(results, outp, pickle.HIGHEST_PROTOCOL)