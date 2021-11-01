# Machine Learning Fall Semester 2021 | Project 1

## (1) Goal

The aim of this project was to develop a machine learning model for predicting the occurrence of the Higgs boson based on data gathered at CERN, Geneva. 

The training dataset consisted in 250000 events and 30 feature columns. Some of these features were raw measurements, others derived parameters.

We were operating under the constraints of sticking to the machine learning techniques that had thus far been introduced in the lecture and not using any libraries except from numpy and plotting libraries.

## (2) Initial considerations and general approach

The methods that had been discussed in the course were:

- Linear regression using gradient descent, stochastic gradient descent or normal equations;
- Ridge regression using normal equations;
- Logistic regression with/without regularization, using gradient descent or stochastic gradient descent.

It soon became obvious to us that it wouldn't be computationally feasible to compare all these different models under all kinds of combinations of hyperparameters and feature extension strategies, although we thought it likely that interdependences would be presence. Therefore, we decided to focus our efforts on two types of models and dig deep into their behaviour unter varying conditions.

The first one involved a Logistic Regression classifier along with polynomial feature extension, trained using gradient descent. The second one is a creative approach, which we named Random Ridge Regression. It uses a large number of ridge regression models trained on a randomly selected subset of features and rows and computes the mean of their weights to generate predictions. By combining a lot of small models, it can reduce the over-fitting and is better than a single random ridge classifier.

## (3) Preprocessing

We performed a grid search to compare different preprocessing approaches, varying NaN treatment strategies (replacement by mean vs. replacement by median vs. replacement by 0 vs. removal of columns with NaN values) and the presence or absence of standardization. The comparison was carried out using a basic logistic regression model with stochastic gradient descent (batch-size 50000) on the original dataset extended with non-interactive terms of degree two, with a learning rate of .01 and sampling the values of the initial weights from a normal distribution with mean = 0 and std = 1. The best validation accuracy (mean over all instances of 4-fold cross-validation for each preprocessing strategy) was achieved when replacing NaN values with the median and standardizing the data. For details on this grid search please refer to `scripts_project/preproc_gridSearch.py` and `scripts_project/ShowAndPlotResults.ipynb`.

To incorporate our observations regarding NaN values (cf. section on data description), we created a one-hot encoding where a new feature is created for each of the features that exhibit NaN values. Individual observations' values on these features are set to 1 when the corresponding original feature is NaN, and to 0 otherwise.

To incorporate our observations regarding NaN values, we created a one-hot encoding where a new feature is created for each of the features that exhibit NaN values. Individual observations' values on these features are set to 1 when the corresponding original feature is NaN, and to 0 otherwise.

## (4) Model tweaking and results: logistic regression

To select the best hyperparameters for logistic regression, we performed a grid search over the cross-product of ranges of values we had selected as appropriate after initial experiments. In particular, we varied:
- batch size: 10000, 20000, 50000, entire dataset (N).
- learning rate: we explored values in the range e-5 to 1.
- distribution from which the values of the initial w-vector were sampled: uniform vs. normal; while the range was always [-1,1].

In picking the best combination we had to exclude those for which we encountered numerical stability problems, mostly associated with large learning rates and identifiable with a non smooth-looking training loss. The best combination was using a batch size of 10000, a learning rate 6.8e-5 and normal initial distribution of weights. 

Thereafter, we proceeded to assess which higher-order terms (i.e., interactive and non-interactive polynomials) to include. We decided to start from a polynomial of degree 3 without interacting terms and to randomly select batches of 10 interacting terms of degree 2 among the ones possible using custom functions for lazy feature generation (please refer to the functions `gen_new_features`, `get_exponents` and `exponents` in `scripts/features_ext.py`). Thus, for these different sets of interaction terms, we trained our logistic regression using different combinations of interacting terms. The best performance achieves 81% validation mean accuracy (averaged over 4-fold cross-validation).

## (5) Model tweaking and results: Random Ridge Classifier

This classifier uses a lot of parameters so it is long to train and to generate predictions. Instead of performing a grid search, we manually selected some parameters to build a reference model and evaluated each of them by splitting the data in 70% for the training set and 30% for the validation set. We then tried to modify each parameter to observe the effect on the accuracy. 

Overall, Random Ridge Regression Classifier performed better than logistic regression. We chose this model for our final prediction with a polynomial extension of degree 9, 50000 rows per classifier, all the features and 10 classifiers. It reached an accuracy of 82.8% with a F1 score of 74.2. This result is very close to what we obtained in our validation steps. As a reference, the best we could have with a single ridge regression model was 82.1%.
