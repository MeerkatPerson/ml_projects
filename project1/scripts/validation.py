import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions to make all kind of validation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Simply splitting the data
def split_data(x, y, ratio, verbose = False):
    """
    Function to split the data in training and validation set.
    Shuffling of the original dataset is executed
    Arguments:
        -x: independent variable of the dataset
        -y: dependent variable of the dataset
        -ratio: the size ratio training/validation
        -verbose: prints the number of samples per subset
    Returns:
        - (sub_x_1, sub_y_1, sub_x_2, sub_y_2): tuple of subsets
    """
    n_samples = y.shape[0]

    indices = np.random.permutation(n_samples)
    sub_x_1 = x[indices][:int(ratio*n_samples)]
    sub_x_2 = x[indices][int(ratio*n_samples):]

    sub_y_1 = y[indices][:int(ratio*n_samples)]
    sub_y_2 = y[indices][int(ratio*n_samples):]

    if verbose:
        print('ration:\t', ratio)
        print('ratio of y == 1 1st subset:\t', np.round_((sub_y_1 == 1).sum()/(ratio*n_samples), decimals=2))
        print('ratio of y == 1 2nd subset:\t', np.round_((sub_y_2 == 1).sum()/((1-ratio)*n_samples), decimals=2))
    return sub_x_1, sub_y_1, sub_x_2, sub_y_2