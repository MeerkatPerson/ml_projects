from copy import Error
import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STANDARDIZATION, ONE HOT ENCODING, NAN TREATMENT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def standardize(x):
  """Standardize the original data set."""
  mean = x.mean(axis = 0)
  std = x.std(axis = 0)
  standardized = (x-mean)/ std 
  return standardized, mean, std

def one_hot_encoding(tx_train):
  """
      For each columns of tx_train which has a NaN, generates an array L where
      L[i] = 1 if tx_train[i] is a NaN else 0.
      Param : 
        - tx_train : N*D 
      Hypothesis :
        - A NaN in tx_train is a value less than -998.
      
  """
  NANVAL = -998.
  tx = tx_train
  tx = np.where(tx < NANVAL, np.NaN, tx)
  res = np.empty((tx.shape[0],1))
  for i in range(tx.shape[1]):
    tmp = np.isnan(tx[:, i])
    if tmp.any():
      res = np.column_stack((res, tmp))
  return res[:, 1:]


def preprocess(y, tx, nan_strategy, standardize_=True, onehotencoding=False):
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
      - onehotencoding : onehotencodes the data
  Returns : 
      -(res_y, res_x) : tuple of processed data
  """
  NANVAL = -998
  
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

  if standardize_: 
    res_x, _, _ = standardize(res_x)
  if onehotencoding:
    res_x = np.column_stack((res_x, one_hot_encoding(tx)))
  return res_y, res_x


