import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This module contains methods to make feature extentions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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