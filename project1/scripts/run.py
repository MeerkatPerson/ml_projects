import numpy as np 
from proj1_helpers import *
from preprocessing import *
from features_ext import *
from classifiers import *

TRAIN = '../data/train.csv'
TEST = '../data/test.csv'
OUTPUT = './submission.csv'
NANVAL = -998
SEED = 0
DEGREE = 9


np.set_printoptions(formatter={'float_kind':'{:f}'.format})


def main():
    np.random.seed(SEED)
    print("Loading train data")
    y_tr, tx_tr, _ = load_csv_data(TRAIN)
    print("Loading test data")
    y_te, tx_te, ids_test = load_csv_data(TEST)
    print("preprocessing train data")
    y_tr,tx_tr = preprocess(y_tr, tx_tr, 'NanToMean', onehotencoding=True)
    print("preprocessing testing data")
    y_te,tx_te = preprocess(y_te, tx_te, 'NanToMean', onehotencoding=True)
    del y_te
    print("building centroids")
    centroids = build_centroids(y_tr, tx_tr)
    centroids = np.asarray([b for a,b in centroids])
    print("extending the features with interactions terms and monomials for training set ~ 5 minutes on google colab ")
    additionnal_functions=[]
    
    tx_tr, d = build_poly_interaction(tx_tr, DEGREE, additionnal_functions, centroids)
    print("training")

    # Building the classifier
    n_classifier = 10
    lambda_ = 0.01
    number_of_rows = 50000
    features_per_classifier = 41
    use_centroids = True
    clf = ClassifierRandomRidgeRegression(n_classifier, lambda_, number_of_rows, features_per_classifier, use_centroids)
    # Training
    clf.train(y_tr, tx_tr, d)
    preds_tr = clf.predict(tx_tr)
    print("Training accuracy : {}".format(clf.accuracy(preds_tr, y_tr)))
    del y_tr
    del tx_tr
    del preds_tr
    print("predicting ~ 20 minutes on google colab")
    preds_test = []
    for i in range(tx_te.shape[0]//100000 + 1):
        print("predicted = {}".format(i*100000))
        x = tx_te[100000*i:100000*(i+1),]
        x, _ = build_poly_interaction(x, DEGREE, additionnal_functions, centroids)
        preds_test = preds_test + list(clf.predict(x))
    print ("writing submission file")
    create_csv_submission(ids_test, preds_test, OUTPUT)
    print("file written")


if __name__ == '__main__':
    main()