import h5py
import numpy as np
import sys

fname = sys.argv[1]
print(fname)
f = h5py.File(fname, 'r')
mjj = f['mjj'][()]
m_low = 3050
m_high = 3250
mask = (mjj > m_low) & (mjj < m_high)
count = np.sum(mask)

print("%i events between %i and %i GeV" % (count, m_low, m_high))

