# Machine Learning Fall Semester 2021 | Project 1

## (1) Goal

The aim of this project was to develop a machine learning model for predicting the occurrence of the Higgs boson based on data gathered at CERN, Geneva. 

The training dataset consisted in 250000 events and 30 feature columns. Some of these features were raw measurements, others derived parameters.

We were operating under the constraints of sticking to the machine learning techniques that had thus far been introduced in the lecture and not using any libraries except from numpy and plotting libraries.

## (2) Initial considerations and general approach

Our toolkit consisted in:

- Linear regression using gradient descent, stochastic gradient descent or normal equations;
- Ridge regression using normal equations;
- Logistic regression with/without regularization, using gradient descent or stochastic gradient descent.

It soon became obvious to us that it wouldn't be computationally feasible to compare all these different models under all kinds of combinations of hyperparameters and feature extension strategies. Therefore, we decided to focus our efforts on two types of models and dig deep into their behaviour unter varying conditions.

As we wanted to predict binary data (Higgs bosom present or absent), logistic regression seemed the obvious choice. Still, we were curious to observe the behaviour of logistic regression in comparison to another model. Hence, we chose to investigate ridge regression as well. 

## (3) Preprocessing

We performed a grid search to compare different preprocessing approaches, varying NaN treatment strategies (replacement by mean vs. replacement by median vs. replacement by 0 vs. removal of columns with NaN values) and the presence or absence of standardization. The comparison was carried out using a basic logistic regression model with non-stochastic gradient descent on the original dataset extended with non-interactive terms of degree two, with a learning rate of .01 and sampling the values of the initial weights from a normal distribution with mean = 0 and sd = 1. The best test accuracy (mean over all instances of 4-fold cross-validation for each preprocessing strategy) was achieved when replacing NaN values with the median and standardizing the data. For details on this grid search please refer to `scripts_project/ShowAndPlotResults.ipynb`.

To incorporate our observations regarding NaN values (cf. section on data description), we created a one-hot encoding where a new feature is created for each of the features that exhibit NaN values. Individual observations' values on these features are set to  1 when the corresponding original feature is NaN, and to 0 otherwise.

## (4) Hyperparameter selection

### Logistic regression

To select the best hyperparameters, we performed a grid search over the cross-product of ranges of values we had selected as appropriate after initial experiments. In particular, we varied:
- dataset: to the original dataset (unaltered except for the adjustments described under 'Preprocessing'), we iteratively added: non-interactive higher-order terms of degree 2; random batches of interactive terms of degree 2; non-interactive higher-order terms of degree 3; random batches of interactive terms of degree 3. (TODO: is that all?)
- batch size: TODO: add values we ended up trying
- learning rate: TODO: add values we ended up trying
- distribution from which the values of the initial w-vector were sampled: uniform vs. normal vs. logarithmic; while the range was always [-1,1].
- penalization: TODO: did we end up doing that after all or can we think of a good reason why we didn't? 

While we pondered including other functions of the original features (log, exp, sqrt, ...), we later realized that when using standardization and hence obtaining values in an interval [-1,1], any of the elementary functions can be approximated by a Taylor expansion, and hence decided that polynomial extension would therefore suffice.

### Ridge Regression

TODO: describe model selection strategy

## (5) Results

TODO: WHAT ARE THEY ?????

## (6) Conclusion

TODO: Depends on results I'd say ....