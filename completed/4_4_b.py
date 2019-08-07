from math import e, log, ceil
import numpy as np
from numba import jit
from time import time
from math import inf
import shelve
import matplotlib.pyplot as pl
pl.close('all')

@jit
def hash_fcn(a, b, x): # p, n_buckets
    return ((a * (x % p) + b) % p) % numBuckets

delta = e ** (-5)
eps = e * 1e-4
p = 123457
aa = []
bb = []
numHashes = ceil(log(1 / delta))
numBuckets = ceil(e / eps)
#fileSuffix = '_tiny'
fileSuffix = ''

with open('./data/hash_params.txt') as f:
    for line in f:
        a, b = line.split()
        aa.append(int(a))
        bb.append(int(b))
assert len(aa) == numHashes



c = np.zeros((numHashes, numBuckets), dtype=int)

t0 = time()
with open('./data/words_stream' + fileSuffix + '.txt') as f:
    for i, wordID in enumerate(f):
        if i % 100000 == 0:
            print(i)
        wordID = int(wordID)
        for j in range(numHashes):
            c[j, hash_fcn(aa[j], bb[j], wordID)] += 1
t = i + 1
print(time() - t0)

realCounts = {}
with open('./data/counts' + fileSuffix + '.txt') as f:
    for line in f:
        wordID, cnt = line.split()
        realCounts[int(wordID)] = int(cnt)

hashCounts = {}
for wordID in realCounts:
    hashCount = inf
    for j in range(numHashes):
        cjh = c[j, hash_fcn(aa[j], bb[j], wordID)]
        hashCount = min(hashCount, cjh)
    hashCounts[wordID] = hashCount

error = []
exact = []
for wordID, realCount in realCounts.items():
    error.append((hashCounts[wordID] - realCount) / realCount)
    exact.append(realCount / t)
pl.loglog(exact, error, '.')


my_shelf = shelve.open('allVar' + fileSuffix, 'n') # 'n' for new
for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except: # TypeError:
        # __builtins__, my_shelf, and imported modules can not be shelved.
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()


