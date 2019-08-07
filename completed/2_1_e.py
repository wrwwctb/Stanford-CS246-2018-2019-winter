import scipy.linalg as sl
import numpy as np
M = [[1, 2],
     [2, 1],
     [3, 4],
     [4, 3]]
M = np.array(M)
U, s, Vh = sl.svd(M, full_matrices=False)
print('U')
print(U)
print('Sigma')
print(s)
print('VT')
print(Vh)

Evals, Evecs = sl.eigh(M.T @ M)

indices = np.argsort(Evals)[::-1]

Evals = Evals[indices]
Evecs = Evecs[:, indices]

print('Evals')
print(Evals)
print('Evecs')
print(Evecs)

