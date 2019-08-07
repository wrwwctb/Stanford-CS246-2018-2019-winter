import matplotlib.pyplot as pl
import pickle
pl.close('all')

with open('dist_l1__c1.txt', 'rb') as fh:
    l1c1 = pickle.load(fh)
with open('dist_l1__c2.txt', 'rb') as fh:
    l1c2 = pickle.load(fh)
with open('dist_l22__c1.txt', 'rb') as fh:
    l22c1 = pickle.load(fh)
with open('dist_l22__c2.txt', 'rb') as fh:
    l22c2 = pickle.load(fh)


pl.plot(l1c1, label='L1, c1')
pl.plot(l1c2, label='L1, c2')
pl.legend()

pl.figure()
pl.plot(l22c1, label='L2^2, c1')
pl.plot(l22c2, label='L2^2, c2')
pl.legend()

def change(L):
    return (L[0] - L[10]) / L[0]

print('L2^2 c1', change(l22c1))
print('L2^2 c2', change(l22c2))
print('L1   c1', change(l1c1))
print('L1   c2', change(l1c2))
