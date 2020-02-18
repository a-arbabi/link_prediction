import numpy as np

def train_model(train_matrix, lambda_param):
  g = np.matmul(train_matrix.T, train_matrix)
  diag_indecies = np.diag_indices(g.shape[0])
  g[diag_indecies] += lambda_param
  p = np.linalg.inv(g)
  b = p/-(np.diag(p))
  
  b[diag_indecies] = 0
  ans = np.matmul(train_matrix, b)
  return ans, b