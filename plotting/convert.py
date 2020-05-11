import os,sys
sys.path.append('..')
import h5py
import pandas as pd
from utils.rootconvert import to_root

h5File = h5py.File('small.h5','r')
treeArray = h5File['test'][()]
lColumns = ['is_signal','Mjj','Mj1','Mj2',
            'j1_score', 'j2_score' ]
df = pd.DataFrame(treeArray,columns=lColumns)
h5File.close()

to_root(df,'small.root',key="Events")

