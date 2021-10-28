from copy import Error
import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STANDARDIZATION, NAN TREATMENT, OUTLIER REMOVAL
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

    

def preprocess(y, tx, nan_strategy, standardize_=True, outliers = False):
  """
  Do the preprocessing of the data.
  Argument:
      - y : of shape (N, )
      - tx: of shape (N, D)
      - nan_strategy: string. Defines how we handle the data. One of the following:
          1. 'NanToMean', replaces NaNs with the mean
          2. 'NanToMedian', replaces the NaNs with the median
          3. 'RemoveNan', removes the rows containing the NaNs
          4. 'RemoveNanFeatures' removes the columns with NaNs
          5. 'NanTo0', replaces the NaNs with 0
      - standardize: standardizes the data
      - outliers TODO
  Returns : 
      -(res_y, res_x) : tuple of processed data
  """
  NANVAL = -998
  
  #TODO : outliers
  res_x = tx
  res_y = y
  res_x = np.where(res_x < NANVAL, np.NaN, res_x)

  indices = np.where(np.isnan(res_x))
  if nan_strategy=='NanToMean':
    # Replace with mean
    means = np.nanmean(res_x, axis=0)
    res_x[indices] = np.take(means, indices[1]) 
  elif nan_strategy=='NanToMedian':
    # Replace with median
    medians = np.nanmedian(res_x, axis=0)
    res_x[indices] = np.take(medians, indices[1])
  elif nan_strategy=='RemoveNan':
    # Remove the NaN
    rows_with_nan = ~np.isnan(res_x).any(axis=1)
    res_y, res_x = res_y[rows_with_nan], res_x[rows_with_nan]
  elif nan_strategy=='RemoveNanFeatures':
    # Remove columns with NaN
    columns_with_nan = ~np.isnan(res_x).any(axis=0)
    res_x = res_x[:,columns_with_nan]
  elif nan_strategy == 'OnlyNanFeatures':
    columns_wo_nan = np.isnan(res_x).any(axis=0)
    res_x = res_x[:,columns_wo_nan]
    rows_with_nan = ~np.isnan(res_x).any(axis=1)
    res_y, res_x = res_y[rows_with_nan], res_x[rows_with_nan]
  elif nan_strategy== 'NanTo0':
    # Replace with 0
    res_x = np.nan_to_num(res_x)
  else:
    raise Error('specify a correct strategy')


  if outliers : 
    #TODO remove outliers
    pass
  if standardize_: 
    res_x, _, _ = standardize(res_x)
  return res_y, res_x


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# F U N C T I O N S  R E L A T E D  T O  F E A T U R E  E X T E N S I O N
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

def build_poly_standard(x, degree):
    
    powers = []
    
    for col in x.T:
    
      for j in range(0, degree+1):
        
        powers.append((col ** j).reshape(-1,1)) 
        
    return np.concatenate(powers,axis=1)

def build_poly(x, degree, functions, centroids):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
        also applies functions
    """
    poly = np.ones((len(x), 1))

    for i in range(x.shape[1]):
        for deg in range(1, degree+1):
            poly = np.c_[poly, np.power(x[:, i], deg)]
        for f in functions:
            poly = np.c_[poly, f(x[:, i])]
    for c in centroids:
        poly = np.c_[poly, kernel(x,c)]

    return poly

def build_poly_interaction(x, degree, functions, centroids):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
        also applies functions
    """
    poly = np.ones((len(x), 1))
    for i in range(x.shape[1]):
        for deg in range(1, degree+1):
            poly = np.c_[poly, np.power(x[:, i], deg)]
        for f in functions:
            poly = np.c_[poly, f(x[:, i])]
        for j in range(i+1, x.shape[1]):
            poly = np.c_[poly, x[:,i] * x[:,j]]
            
    for c in centroids:
        poly = np.c_[poly, kernel(x,c)]

    return poly