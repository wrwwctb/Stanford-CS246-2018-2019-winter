import numpy as np
import matplotlib.pyplot as pl
pl.close('all')
dataPath = './data/ratings.train.txt'

# check ranges of i and u
#with open(dataPath) as f:
#    mi = mu = float('inf')
#    Mi = Mu = -float('inf')
#    for line in f:
#        i, u, R = map(int, line.split())
#        mi = min(mi, i)
#        mu = min(mu, u)
#        Mi = max(Mi, i)
#        Mu = max(Mu, u)

m = 943
n = 1682

k = 20
reg = .1
numIter = 40
lr = .01

np.random.seed(0)

Q = np.random.random((m, k)) * np.sqrt(5 / k)
P = np.random.random((n, k)) * np.sqrt(5 / k)
losses = []

fig = pl.figure()

for it in range(numIter):
    print('iter %d in progress' % it)

    with open(dataPath) as f:
        for line in f:
            i, u, R = map(int, line.split())
            i -= 1
            u -= 1

            R_QPT = R - Q[i] @ P[u].T
            dQi = lr * 2 * (R_QPT * P[u] - reg * Q[i])
            dPu = lr * 2 * (R_QPT * Q[i] - reg * P[u])
            Q[i] += dQi
            P[u] += dPu

            # updating separately improves loss by 1%
            #P[u] += lr * 2 * ((R - Q[i] @ P[u].T) * Q[i] - reg * P[u])
            #Q[i] += lr * 2 * ((R - Q[i] @ P[u].T) * P[u] - reg * Q[i])


    # calc loss per epoch
    # collecting components of loss during update above would be wrong
    loss = reg * (np.linalg.norm(P) ** 2 + np.linalg.norm(Q) ** 2)
    with open(dataPath) as f:
        for line in f:
            i, u, R = map(int, line.split())
            i -= 1
            u -= 1
            loss += (R - Q[i] @ P[u].T) ** 2
    losses.append(loss)
    fig.clf()
    pl.plot(range(it + 1), losses)
    pl.pause(.01)

pl.xlabel('Epoch')
pl.ylabel('Loss')
