import shelve
import matplotlib.pyplot as pl
pl.close('all')

fileSuffix = ''

my_shelf = shelve.open('allVar' + fileSuffix)
error = my_shelf['error']
exact = my_shelf['exact']
my_shelf.close()
m_error = min(error)
M_error = max(error)
m_exact = min(exact)
M_exact = max(exact)

pl.loglog(exact, error, '.', markersize=.5)
pl.loglog([m_exact, M_exact], [1, 1], ':', color='xkcd:gray')
pl.loglog([1e-5, 1e-5], [m_error, M_error], ':', color='xkcd:gray')
pl.xlabel('exact word frequency')
pl.ylabel('relative error')