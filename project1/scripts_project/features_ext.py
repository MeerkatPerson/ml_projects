import numpy as np
from preprocessing import standardize

from collections import defaultdict

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This module contains methods to make feature extentions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# ------------------------------------------------------------------------------------------------------------------
# (a) CENTROIDS & KERNEL
# ------------------------------------------------------------------------------------------------------------------

def build_centroids(y, x):
    res = []
    for cl in set(y):
      res.append((cl, np.mean(x[y==cl], axis = 0)))
    return res
    # [centroid class -1, centroid class 1]

def kernel(x, centroid):
    return np.exp(-np.linalg.norm(x-centroid, axis = 1)**2)

# ------------------------------------------------------------------------------------------------------------------
# (b) POLYNOMIAL EXTENSION
# ------------------------------------------------------------------------------------------------------------------

def build_poly_and_standardize(x_train_u, x_test_u, degree):
    '''Uses build poly standard to build polynomial from NON-STANDARDIZED data.
    Returns the output
    '''

    #expand the features using the non-standardized data
    x_train_e = build_poly_standard(x_train_u, degree)
    x_test_e = build_poly_standard(x_test_u, degree)

    #standardize the data
    temp, _, _ = standardize(x_train_e[:, 1:])
    x_train_e = np.concatenate(( x_train_e[:, 0].reshape(-1, 1), temp ), axis = 1)

    temp, _, _ = standardize(x_test_e[:, 1:])
    x_test_e = np.concatenate(( x_test_e[:, 0].reshape(-1, 1), temp ), axis = 1)

    return x_train_e, x_test_e


def build_poly_standard(x, degree):
    """
    Build polynomial up to a given degree without interacting terms"""
    #build the constant terms
    expanded = np.ones_like(x[:, 0]).reshape(-1, 1)
    expanded = np.concatenate((expanded, x), axis = 1)

    #if degree smaller than 2 return
    if degree <2:
        return expanded
    #otherwise expand the features
    else:
        for d in range(2, degree +1):
            expanded = np.concatenate(
                (expanded, x**d), axis = 1
            )
        return expanded

def build_poly_interaction(x, degree, functions, centroids, initial_features = 30):
    """
    Polynomial basis functions for input data x, for j=0 up to j=degree. 
    Applies a list of functions to each columns and builds interaction terms. 
    Adds the kernel function between x[i] and the centroid of each class.
    Params :
        - x : N*D after one hot encoding
        - degree : int
        - functions : List[functions] of size K
        - centroids : J*D (in this project J = 2)
        - initial_features : number of features before one hot encoding. It is used to avoid
        duplicating columns.
    Returns :
        extended x with size N * (1 + D*(degree+K+J) + D(D-1)/2)
    """
    d = defaultdict(set)
    res = np.ones((len(x), 1))
    column = 0
    for i in range(x.shape[1]):
        if i < initial_features:
            for deg in range(1, degree+1):
                res = np.column_stack((res, np.float_power(x[:, i], deg))) # x1^2, x1^3...x1^degree
                column += 1
                d[i].add(column)
        for f in functions:
            res = np.column_stack((res, f(x[:, i]))) # 1/x1, x1^1/3
            column += 1
            d[i].add(column)
        for j in range(i+1, x.shape[1]):
            res = np.column_stack((res, x[:,i] * x[:,j])) # x1,x2,x3 -> x1*x2, x2*x3, x1*x3
            column += 1
            d[i].add(column)
            d[j].add(column)
        
    for c in centroids:
        res = np.column_stack((res, kernel(x,c)))

    return res, d

###################################################################################
# General code to get interactive terms of any degree
####################################################################################


def gen_new_features(x, exp):
    """
    Generate the new features, exponentiating x using the indices contained in exp.
    Input: 
        -x: array of shape (n_samples, n_initial_features)
    Returns: 
        -res: array  of shape (n_samples, n_monomials_in_homogeneous_polynomial)"""
    #reshape x to allow broadcasting
    x =  np.expand_dims(x, axis = 1)
    #get the exponents
    #exp = exponents(n_variables=x.shape[-1], degree= degree )
    #compute the new features
    res = x**exp
    res = res.prod(axis = -1)
    return res

def get_exponents(n_variables, degree):
    """
    Generates the eponents for all the monomials for homogeneous polynomial of a given degree
    It's a recursive function.
    Input:
        -n_variables: the number of variables in of the polynomial
        -degree: the desired degree
    Returns:
        -iterable of indices of exponents
    """
    #initial step
    if n_variables == 1:
        yield [degree]
        return
    #recursive step
    for i in range(degree + 1):
        for t in get_exponents(n_variables - 1, degree - i):
            yield [i] + t

def exponents(n_variables, degree, n_features, non_interaction_first=False):
    """ 
    Creates a list of groups of exponents.
    Each group a number "n_features" of exponents referring to monomials in the homogeneous polynomial of a given "degree" and "n_variables" variables
    Arguments:
        - n_variables: the number of variables in the polynomial
        - degree: the desired degree of the homogeneous polynomial
        - n_features:  the number of elements in each group
        - non_interaction_first: if True the first element of the list will the group of exponents associated with "non-interacting" terms
                                e.g. only x_1^2, x_2^2 and not x1*x2
    Returns:
        -exp_sub_list: a list containing groups of exponents. Each group is a np.array
     """
    exp_sub_list = []

    #create a list
    exp = [e for e in get_exponents(n_variables, degree)]
    #cast it to np.array
    exp = np.array(exp)

    #if non_interaction_first required, the first element of the list contains the non interactive terms
    if non_interaction_first:
        #count the number od zeros in a row and build the index
        indices = ((exp == 0.).sum(axis = 1) == (exp.shape[1] - 1.))
        exp_sub_list += [exp[indices].copy()]
        exp = exp[~indices]
        
    #NOTE i added this (27.10.) so we get reproducibility
    np.random.seed(42)

    #make random permutation of the indeces
    indices = np.random.permutation(exp.shape[0])
    exp = exp[indices]
    n_slices = int(exp.shape[0] / n_features)
    exp_sub_list += [exp[i*n_features:(i+1)*n_features] for i in range(n_slices)]
    if n_slices*n_features < exp.shape[0]:
        exp_sub_list += [exp[n_slices*n_features:]]
    return exp_sub_list

def number_monomials(n_variables, degree):
    """
    Computes how many ters has an homogeneous polynomial of a given degree and 
    with a given n_variables
    """
    a = np.math.factorial(n_variables + degree - 1)
    b = np.math.factorial(degree)
    c = np.math.factorial(n_variables - 1)
    return a/(b*c)