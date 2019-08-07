import numpy as np

dataPath = './data/user-shows.txt'
showsPath = './data/shows.txt'

shows = []
with open(showsPath) as f:
    for line in f:
        shows.append(line.strip())

m = 9985
n = 563

R = np.zeros((m, n), dtype=np.int16)
with open(dataPath) as f:
    for i, line in enumerate(f):
        R[i, :] = list(map(int, line.split()))

p = np.sum(R, axis=1)
q = np.sum(R, axis=0)

p_ = 1 / np.sqrt(p)
q_ = 1 / np.sqrt(q)

RTp_ = R.T * p_[None, :]
Rq_ = R * q_[None, :]

rU = RTp_.T @ RTp_ @ R
rI = R @ Rq_.T @ Rq_

alexi = 499
rU_focus = rU[alexi, :100]
rI_focus = rI[alexi, :100]

rU_idx = np.argsort(rU_focus)[::-1][:5]
rI_idx = np.argsort(rI_focus)[::-1][:5]

rU_results = list(zip(rU_focus[rU_idx], rU_idx, [shows[i] for i in rU_idx]))
rI_results = list(zip(rI_focus[rI_idx], rI_idx, [shows[i] for i in rI_idx]))

print(rU_results)
print(rI_results)
