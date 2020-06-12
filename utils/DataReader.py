import h5py
import numpy as np
import math
import tensorflow as tf 
import copy

def expandable_shape(d_shape):
    c_shape = list(d_shape)
    c_shape[0] = None
    c_shape = tuple(c_shape)
    return c_shape


def append_h5(f, name, data):
    prev_size = f[name].shape[0]
    f[name].resize(( prev_size + data.shape[0]), axis=0)
    f[name][prev_size:] = data

class MyGenerator(tf.keras.utils.Sequence):
    def __init__(self, f, n, batch_size, key1, key2):
        self.f = f
        self.n = n
        self.batch_size = batch_size
        self.n_batches = int(np.ceil(self.n / self.batch_size))
        self.idx = 0

        self.key1 = key1
        self.key2 = key2

    def __next__(self):
        if self.idx >= self.n_batches:
           self.idx = 0
        result = self.__getitem__(self.idx)
        self.idx += 1
        return result

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.n_batches

    def on_epoch_end(self):
        pass
        #print("Epoch end")

    def __getitem__(self, i):
        return self.f[self.key1][self.batch_size*i:(i+1)*self.batch_size], self.f[self.key2][i*self.batch_size:(i+1)*self.batch_size]





class DataReader:
    def __init__(self, f_name, signal_idx =1, keys = ['j1_images', 'j2_images', 'mjj'], sig_frac = -1., start = 0, stop = -1, m_low = -1., m_high = -1., val_frac = 0.):
        self.f_name = f_name
        self.signal_idx = signal_idx
        self.keys = keys
        self.sig_frac = sig_frac
        self.start = start
        self.stop = stop
        self.val_frac = val_frac
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
        self.nTrain = 0
        self.nVal = 0

                

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


            #save labels 
            d_labels = labels[mask]
            self.nEvents += d_labels.shape[0]
            if(self.val_frac > 0.):
                n_val = int(np.floor(d_labels.shape[0]*self.val_frac))
                n_train = d_labels.shape[0] - n_val
                t_labels = d_labels[:n_train]
                v_labels = d_labels[n_train:]
            else:
                t_labels = d_labels
            if(i==0):
                self.f_storage.create_dataset('label', data = t_labels, chunks = True, maxshape = expandable_shape(d_labels.shape))
                if(self.val_frac > 0.): self.f_storage.create_dataset('val_label', data = v_labels, chunks = True, maxshape = expandable_shape(d_labels.shape))
            else:
                append_h5(self.f_storage, 'label', t_labels)
                if(self.val_frac > 0.): append_h5(self.f_storage, 'val_label', v_labels)



            #save data in other keys
            data = None
            for ikey,key in enumerate(self.keys):
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
                tdata = None
                vdata = None
                if(self.val_frac > 0.):
                    n_val = int(np.floor(data.shape[0]*self.val_frac))
                    n_train = data.shape[0] - n_val
                    tdata = data[:n_train]
                    vdata = data[n_train:]
                else:
                    tdata = data
                if(ikey == 0):
                    self.nTrain += tdata.shape[0]
                    if(self.val_frac > 0.): self.nVal += vdata.shape[0]
                if(i==0):
                    c_shape = expandable_shape(data.shape)
                    self.f_storage.create_dataset(key, data = tdata, chunks = True, maxshape = c_shape)
                    if(self.val_frac > 0.): self.f_storage.create_dataset("val_"+key, data = vdata, chunks = True, maxshape = c_shape)
                else:
                    append_h5(self.f_storage, key, data)
                    if(self.val_frac > 0.): append_h5(self.f_storage, "val_"+key, data)


        f.close()
        self.keys.append('label')
        print("Kept %i events after selection" % self.nEvents)
        self.ready = True
        if(self.val_frac > 0.):
            print("Training events: %i, Validation events: %i" % (self.nTrain, self.nVal))
            new_keys = copy.copy(self.keys)
            for key in self.keys: new_keys.append("val_" + key)
            self.keys = new_keys

    def make_Y_mjj(self, mjj_low, mjj_high):
        self.keys.append('Y_mjj')
        mjj = self.f_storage['mjj'][()]
        mjj_window = ((mjj > mjj_low) & (mjj < mjj_high))
        self.f_storage.create_dataset('Y_mjj', data = mjj_window)
        del mjj, mjj_window

        if(self.val_frac > 0.):
            self.keys.append('val_Y_mjj')
            mjj = self.f_storage['val_mjj'][()]
            mjj_window = ((mjj > mjj_low) & (mjj < mjj_high))
            self.f_storage.create_dataset('val_Y_mjj', data = mjj_window)
            del mjj, mjj_window



    def __getitem__(self, key):
        if(not self.ready):
            print("Datareader has not loaded data yet! Must call read() first ")
            exit(1)
        if(key not in self.keys):
            print("Key %s not in list of preloaded keys!" % key, self.keys)
            exit(1)

        return self.f_storage[key][()]

    def gen(self, key1, key2, batch_size = 256):
        if(not self.ready):
            print("Datareader has not loaded data yet! Must call read() first ")
            exit(1)
        if(key1 not in self.keys or key2 not in self.keys):
            print("Key %s not in list of preloaded keys!" % key, self.keys)
            exit(1)
        if( (("val" in key1) and ("val" not in key2)) or (("val" in key2) and ("val" not in key1)) ):
            print("Only one of the keys is for validation data? %s, %s " %(key1, key2))
        
        n_objs = self.f_storage[key1].shape[0]
        n_objs2 = self.f_storage[key2].shape[0]

        if(n_objs != n_objs2):
            print("Mismatched datasets size ?? Key %s has size %i, key %s has size %i " %(key1, n_objs, key2, n_objs))


        h5_gen = MyGenerator(self.f_storage, n_objs, batch_size, key1, key2)
        n_batches = int(math.ceil(float(n_objs)/batch_size))



        return h5_gen

    def __del__(self):
        if(self.ready):
            os.system("rm %s" % 'DReader_temp.h5')

    
        

    



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
