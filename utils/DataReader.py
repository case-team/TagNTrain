import h5py
import numpy as np
import math
import tensorflow as tf 

def expandable_shape(d_shape):
    c_shape = list(d_shape)
    c_shape[0] = None
    c_shape = tuple(c_shape)
    return c_shape


def append_h5(f, name, data):
    prev_size = f[name].shape[0]
    f[name].resize(( prev_size + data.shape[0]), axis=0)
    f[name][prev_size:] = data

class h5_gen:
    def __init__(self, f, n, key1, key2):
        self.f = f
        self.n = n
        self.key1 = key1
        self.key2 = key2

    def __call__(self):
        for i in range(self.n):
            yield self.f[self.key1][i], self.f[self.key2][i]


class DataReader:
    def __init__(self, f_name, signal_idx =1, keys = ['j1_images', 'j2_images', 'mjj'], sig_frac = -1., start = 0, stop = -1, m_low = -1., m_high = -1., val_frac = 0.):
        self.f_name = f_name
        self.signal_idx = signal_idx
        self.keys = keys
        self.sig_frac = sig_frac
        self.start = start
        self.stop = stop
        f = h5py.File(f_name, "r")
        if(self.stop == -1): self.stop = f['truth_label'].shape[0]
        f.close()
        self.nEvents_file = self.stop - self.start
        self.m_low = m_low
        self.m_high = m_high

        self.chunk_size = 200000
        self.max_load = 1000000 #max size to load without chunks

        self.ready = False

        self.f_storage = h5py.File("DReader_temp.h5", "w")


    def read(self):
        self.nChunks = 1
        loading_images = False
        for key in self.keys:
            if('image' in key): loading_images = True

        if(self.nEvents_file > self.max_load and loading_images):
            self.nChunks = int(math.ceil(float(self.nEvents_file)/self.chunk_size))

        print("Loading file %s \n" % self.f_name)
        print("Will read %i events in %i chunks" % (self.nEvents_file, self.nChunks))

        self.nEvents = 0
                

        f = h5py.File(self.f_name, "r")
        for i in range(self.nChunks):
            cstart = self.start + i*self.chunk_size
            if(self.nChunks == 1): cstop = self.stop
            else: cstop = min(cstart + self.chunk_size, self.stop)

            #fill this chunk
            raw_labels = f['truth_label'][cstart: cstop]
            labels = np.zeros_like(raw_labels)
            labels[raw_labels == self.signal_idx] = 1
            #only keep some events
            mask = np.squeeze((raw_labels <= 0) | (raw_labels == self.signal_idx)) 
            if(self.sig_frac > 0.): 
                print("Filtering signal from %.4f to %.4f " %(np.mean(labels), self.sig_frac))
                mask_sig = get_signal_mask_rand(labels, self.sig_frac)
                mask = mask & mask_sig
            if(self.m_low > 0. and self.m_high >0.):
                mjj = f["jet_kinematics"][cstart:cstop,0]
                mjj_mask = (mjj > self.m_low) & (mjj < self.m_high)
                mask = mask & mjj_mask

            d_labels = labels[mask]
            self.nEvents += d_labels.shape[0]
            if(i==0):
                self.f_storage.create_dataset('label', data = d_labels, chunks = True, maxshape = expandable_shape(d_labels.shape))
            else:
                append_h5(self.f_storage, 'label', d_labels)



            data = None
            for key in self.keys:
                if(key == 'mjj'):
                    data = f["jet_kinematics"][cstart:cstop][mask,0]
                elif(key == 'j1_images' or key == 'j2_images'):
                    data = np.expand_dims(f[key][cstart:cstop][mask], axis = -1)
                elif(key == 'jj_images'):
                    j1_img = np.expand_dims(f['j1_images'][cstart:cstop][mask], axis = -1)
                    j2_img = np.expand_dims(f['j2_images'][cstart:cstop][mask], axis = -1)
                    data = np.append(j2_img, j1_img, axis = 3)

                else:
                    data = f[key][cstart:cstop][mask]

                #copy this chunk into class data
                if(i==0):
                    c_shape = expandable_shape(data.shape)
                    print(c_shape)
                    self.f_storage.create_dataset(key, data = data, chunks = True, maxshape = c_shape)
                else:
                    expand_h5(self.f_storage, key, data)


        f.close()
        self.keys.append('label')
        print("Kept %i events after selection" % self.nEvents)
        self.ready = True

    def make_Y_mjj(self, mjj_low, mjj_high):
        self.keys.append('Y_mjj')
        mjj = self.f_storage['mjj'][()]
        mjj_window = ((mjj > mjj_low) & (mjj < mjj_high))
        self.f_storage.create_dataset('Y_mjj', data = mjj_window)
        del mjj, mjj_window


    def __getitem__(self, key):
        if(not self.ready):
            print("Datareader has not loaded data yet! Must call read() first ")
            exit(1)
        if(key not in self.keys):
            print("Key %s not in list of preloaded keys!" % key, self.keys)
            exit(1)

        return self.f_storage[key][()]

    def gen(self, key1, key2):
        if(not self.ready):
            print("Datareader has not loaded data yet! Must call read() first ")
            exit(1)
        if(key1 not in self.keys or key2 not in self.keys):
            print("Key %s not in list of preloaded keys!" % key, self.keys)
            exit(1)
        
        ds = tf.data.Dataset.from_generator(h5_gen(self.f_storage, self.nEvents, key1, key2),
            output_types = (self.f_storage[key1].dtype, self.f_storage[key2].dtype),
            output_shapes = (self.f_storage[key1].shape, self.f_storage[key2].shape))

        return ds
    
        

    



#helper to read h5py mock-datasets
def prepare_dataset(fin, signal_idx =1, keys = ['j1_images', 'j2_images', 'mjj'], sig_frac = -1., start = 0, stop = -1):
    if(stop > 0): print("Selecting events %i to %i \n" % (start, stop))
    data = dict()
    f = h5py.File(fin, "r")
    print("Loading file %s (contains %i events) \n" % (fin, f['truth_label'].shape[0]))
    if(stop == -1): stop = f['truth_label'].shape[0]
    raw_labels = f['truth_label'][start: stop]
    n_imgs = stop - start
    labels = np.zeros_like(raw_labels)
    labels[raw_labels == signal_idx] = 1
    mask = np.squeeze((raw_labels <= 0) | (raw_labels == signal_idx))
    if(sig_frac > 0.): 
        mask0 = get_signal_mask_rand(labels, sig_frac)
        mask = mask & mask0

    for key in keys:
        if(key == 'mjj'):
            data['mjj'] = f["jet_kinematics"][start:stop][mask,0]
        elif('image' in key):
            data[key] = np.expand_dims(f[key][start:stop][mask], axis = -1)
        else:
            data[key] = f[key][start:stop][mask]
    data['label'] = labels[mask]
    print("Signal fraction is %.4f  "  % np.mean(data['label']))
    f.close()
    return data

#create a mask that removes signal events to enforce a given fraction
#removes signal from later events (should shuffle after)
def get_signal_mask(events, sig_frac):

    num_events = events.shape[0]
    cur_frac =  np.mean(events)
    keep_frac = (sig_frac/cur_frac)
    progs = np.cumsum(events)/(num_events * cur_frac)
    keep_idxs = (events.reshape(num_events) == 0) | (progs < keep_frac)
    return keep_idxs


#create a mask that removes signal events to enforce a given fraction
#Keeps signal randomly distributed but has more noise
def get_signal_mask_rand(events, sig_frac, seed=12345):

    np.random.seed(seed)
    num_events = events.shape[0]
    cur_frac =  np.mean(events)
    drop_frac = (1. - (1./cur_frac) * sig_frac)
    rands = np.random.random(num_events)
    keep_idxs = (events.reshape(num_events) == 0) | (rands > drop_frac)
    return keep_idxs
