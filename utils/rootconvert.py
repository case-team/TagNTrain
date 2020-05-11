import numpy as np
import h5py
import pandas as pd
import ROOT

def to_root(df, path, key='my_ttree', mode='w', store_index=True, *args, **kwargs):
    """
    Write DataFrame to a ROOT file.
    Parameters
    ----------
    path: string
        File path to new ROOT file (will be overwritten)
    key: string
        Name of tree that the DataFrame will be saved as
    mode: string, {'w', 'a'}
        Mode that the file should be opened in (default: 'w')
    store_index: bool (optional, default: True)
        Whether the index of the DataFrame should be stored as
        an __index__* branch in the tree
    Notes
    -----
    Further *args and *kwargs are passed to root_numpy's array2root.
    >>> df = DataFrame({'x': [1,2,3], 'y': [4,5,6]})
    >>> df.to_root('test.root')
    The DataFrame index will be saved as a branch called '__index__*',
    where * is the name of the index in the original DataFrame
    """

    if mode == 'a':
        mode = 'update'
    elif mode == 'w':
        mode = 'recreate'
    else:
        raise ValueError('Unknown mode: {}. Must be "a" or "w".'.format(mode))

    from root_numpy import array2tree
    # We don't want to modify the user's DataFrame here, so we make a shallow copy
    df_ = df.copy(deep=False)

    if store_index:
        name = df_.index.name
        if name is None:
            # Handle the case where the index has no name
            name = ''
        df_['__index__' + name] = df_.index

    # Convert categorical columns into something root_numpy can serialise
    for col in df_.select_dtypes(['category']).columns:
        name_components = ['__rpCaT', col, str(df_[col].cat.ordered)]
        name_components.extend(df_[col].cat.categories)
        if ['*' not in c for c in name_components]:
            sep = '*'
        else:
            raise ValueError('Unable to find suitable separator for columns')
        df_[col] = df_[col].cat.codes
        df_.rename(index=str, columns={col: sep.join(name_components)}, inplace=True)

    arr = df_.to_records(index=False)

    root_file = ROOT.TFile.Open(path, mode)
    if not root_file:
        raise IOError("cannot open file {0}".format(path))
    if not root_file.IsWritable():
        raise IOError("file {0} is not writable".format(path))

    # Navigate to the requested directory
    open_dirs = [root_file]
    for dir_name in key.split('/')[:-1]:
        current_dir = open_dirs[-1].Get(dir_name)
        if not current_dir:
            current_dir = open_dirs[-1].mkdir(dir_name)
        current_dir.cd()
        open_dirs.append(current_dir)

    # The key is now just the top component
    key = key.split('/')[-1]

    # If a tree with that name exists, we want to update it
    tree = open_dirs[-1].Get(key)
    if not tree:
        tree = None
    tree = array2tree(arr, name=key, tree=tree)
    tree.Write(key, ROOT.TFile.kOverwrite)
    root_file.Close()

if __name__ == "__main__":
    lFile = h5py.File('/tmp/pharris/Output_pu.h5','r')
    lArr = np.array(lFile.get('Events')[:])
    df = pd.DataFrame(data=lArr)
    to_root(df,'test.root')
