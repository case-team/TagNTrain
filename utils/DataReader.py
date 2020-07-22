import h5py
import numpy as np
import math
import tensorflow as tf 
import copy
import os

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
    def __init__(self, f, n, batch_size, key1, key2, mask = None):
        self.f = f
        self.n = n
        self.batch_size = batch_size
        self.n_batches = int(np.ceil(self.n / self.batch_size))
        self.idx = 0

        self.key1 = key1
        self.key2 = key2
        self.mask = mask

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
        if(self.mask is not None and self.mask.shape[0] == 0):
            return self.f[self.key1][self.batch_size*i:(i+1)*self.batch_size], self.f[self.key2][i*self.batch_size:(i+1)*self.batch_size]
        else:
            mask_local = self.mask[self.batch_size*i:(i+1)*self.batch_size]
            return self.f[self.key1][self.batch_size*i:(i+1)*self.batch_size][mask_local], self.f[self.key2][i*self.batch_size:(i+1)*self.batch_size][mask_local]






class DataReader:
    DR_count = 0
    def __init__(self, f_name, signal_idx =1, keys = None, sig_frac = -1., start = 0, stop = -1, m_low = -1., m_high = -1., val_frac = 0.):
        if(keys == None):
            self.keys = ['j1_images', 'j2_images', 'mjj']
        else:
            self.keys = copy.copy(keys)
        self.f_name = f_name
        self.signal_idx = signal_idx
        self.sig_frac = sig_frac
        self.start = start
        self.stop = stop
        self.val_frac = val_frac
        f = h5py.File(f_name, "r")
        if(self.stop == -1): self.stop = f['event_info'].shape[0]
        f.close()
        self.nEvents_file = self.stop - self.start
        self.m_low = m_low
        self.m_high = m_high

        self.chunk_size = 200000
        self.max_load = 1000000 #max size to load without chunks

        self.ready = False

        self.my_DR_idx = DataReader.DR_count
        self.f_storage = h5py.File("DReader%i_temp.h5" % DataReader.DR_count, "w")
        DataReader.DR_count += 1

        self.mask = np.array([])
        self.val_mask = np.array([])


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
            if(self.signal_idx > 0):
                labels[raw_labels == self.signal_idx] = 1
                mask = np.squeeze((raw_labels <= 0) | (raw_labels == self.signal_idx)) 
            else:
                mask = np.squeeze(raw_labels >= -999999)

            #only keep some events
            cur_sig = np.mean(labels)
            do_filter = self.sig_frac > 0. and cur_sig > self.sig_frac
            print("Signal fraction is %.4f and we want %.4f: Filter %i" %(cur_sig, self.sig_frac, do_filter))
            if(do_filter): 
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
                    append_h5(self.f_storage, key, tdata)
                    if(self.val_frac > 0.): append_h5(self.f_storage, "val_"+key, vdata)


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


    def make_Y_TNT(self, sig_region_cut = 0.9, bkg_region_cut = 0.2, cut_var = np.array([]), mjj_low = -999999., mjj_high = 9999999., sig_high = True):

        if(cut_var.size == 0):
            raise TypeError('Must supply cut_var argument!')

        #sig_high is whether signal lives at > cut value or < cut value
        if(sig_high):
            sig_cut = cut_var > sig_region_cut
            bkg_cut = cut_var < bkg_region_cut
        else:
            sig_cut = cut_var < sig_region_cut
            bkg_cut = cut_var > bkg_region_cut


        if(mjj_low > 0. and mjj_high < 10000.):
            mjj = self.f_storage['mjj'][()]
            mjj_window = ((mjj > mjj_low) & (mjj < mjj_high))
            mjj_sb = ((mjj < mjj_low) | (mjj > mjj_high))
            sig_cut = sig_cut & mjj_window
            bkg_cut = bkg_cut & mjj_sb

        #mjj = self.f_storage['mjj'][()]
        #sig_cut = ((mjj > mjj_low) & (mjj < mjj_high))




        keep_mask = sig_cut | bkg_cut

        Y_TNT = np.zeros_like(cut_var)

        Y_TNT[bkg_cut] = 0
        Y_TNT[sig_cut] = 1
        self.f_storage.create_dataset('Y_TNT', data = Y_TNT)

        self.keys.append("Y_TNT")
        self.apply_mask(keep_mask)


    def __getitem__(self, key):
        if(not self.ready):
            print("Datareader has not loaded data yet! Must call read() first ")
            exit(1)
        if(key not in self.keys):
            print("Key %s not in list of preloaded keys!" % key, self.keys)
            exit(1)
        if('val' in key):
            mask_ = self.val_mask
        else:
            mask_ = self.mask

        if(mask_.shape[0] == 0):
            return self.f_storage[key][()]
        else:
            return self.f_storage[key][()][mask_]

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

        if('val' in key1):
            mask_ = self.val_mask
        else:
            mask_ = self.mask


        h5_gen = MyGenerator(self.f_storage, n_objs, batch_size, key1, key2, mask = mask_)
        return h5_gen

    def labeler_scores(self, model, key, chunk_size = 10000):
        
        n_objs = self.f_storage[key].shape[0]
        n_chunks = int(np.ceil(n_objs / chunk_size))
        results = np.array([])
        if('val' in key):
            mask_ = self.val_mask
        else:
            mask_ = self.mask

        for i in range(n_chunks):
            imgs = self.f_storage[key][chunk_size*i:(i+1)*chunk_size]
            if(mask_.shape[0] != 0):
                local_mask = mask_[chunk_size*i:(i+1)*chunk_size]
                imgs = imgs[mask_]
            preds = model.predict(imgs, batch_size = 512)


            if(len(preds.shape) > 2 ): #autoencoder
                scores = np.mean(np.square(imgs - preds), axis = (1,2)).reshape(-1)

            else: 
                scores = preds.reshape(-1)

            results = np.append(results, scores)


        return results

    #apply a mask to the dataset
    def apply_mask(self, mask, to_training = True):
        if(to_training):
            if(mask.shape[0] != self.nTrain):
                print("Error: Mask shape and number of training events incompatable", mask.shape, self.nTrain)
                exit(1)

            if(self.mask.shape[0] == 0):
                self.mask = mask
            else:
                self.mask = self.mask & mask
            filter_frac = np.mean(mask)
            self.nTrain = int(self.nTrain * filter_frac)
            print("applied mask. Eff %.3f " % filter_frac)
        else: #to val
            if(mask.shape[0] != self.nVal):
                print("Error: Mask shape and number of validation events incompatable", mask.shape, self.nVal)
                exit(1)

            if(self.val_mask.shape[0] == 0):
                self.val_mask = mask
            else:
                self.val_mask = self.val_mask & mask
            filter_frac = np.mean(mask)
            self.nVal = int(self.nVal * filter_frac)

    



    def __del__(self):
        if(self.ready):
            os.system("rm %s" % 'DReader%i_temp.h5' % self.my_DR_idx)

    
        

    



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
    if(cur_frac <= sig_frac):
        return np.ones_like(events)

    drop_frac = (1. - (1./cur_frac) * sig_frac)
    rands = np.random.random(num_events)
    keep_idxs = (events.reshape(num_events) == 0) | (rands > drop_frac)
    return keep_idxs
