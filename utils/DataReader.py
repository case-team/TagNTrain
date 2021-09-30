import h5py
import numpy as np
import math
import tensorflow as tf 
import copy
import os
from .PlotUtils import *

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
    def __init__(self, f, n, batch_size, key1, key2, key3, mask = None):
        self.f = [f]
        self.n = [n]
        self.batch_size = batch_size
        self.f_stops = [int(np.ceil(n / self.batch_size))]
        self.n_total_batches = self.f_stops[0]
        self.masks = [mask]

        if(mask is not None): self.nTotal = np.sum(mask)
        else: self.nTotal = n
        self.idx = 0

        self.key1 = [key1]
        self.key2 = [key2]
        self.key3 = [key3]


    def add_dataset(self, key1,key2, key3, f2 = None, n2 = None, mask = None, dataset = None):
        if(dataset is not None):

            n2 = dataset.f_storage[key1].shape[0]
            mask = dataset.mask
            f2 = dataset.f_storage

        self.f.append(f2)
        self.n.append(n2)
        if(mask is not None): self.nTotal += np.sum(mask)
        else: self.nTotal += n2
        n_batches = int(np.ceil(n2 / self.batch_size))
        self.n_total_batches += n_batches
        self.f_stops.append(self.f_stops[-1] + n_batches)
        self.masks.append(mask)
        self.key1.append(key1)
        self.key2.append(key2)
        self.key3.append(key3)
        
        print("Added dataset with %i batches after batch %i. Now %i total batches" % (n_batches, self.f_stops[-2], self.n_total_batches))
        print(self.f_stops)


    def __next__(self):
        if self.idx >= self.n_total_batches:
           self.idx = 0
        result = self.__getitem__(self.idx)
        self.idx += 1
        return result

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.n_total_batches

    def on_epoch_end(self):
        pass
        #print("Epoch end")

    def __getitem__(self, i):
        f_idx = 0
        f_batch_i = i
        for idx, f_stop in enumerate(self.f_stops):
            if(i >= f_stop):
                f_idx = idx+1
                f_batch_i = i - f_stop

        f_idx = f_idx % len(self.f)
        f = self.f[f_idx]
        mask = self.masks[f_idx]
        key1 = self.key1[f_idx]
        key2 = self.key2[f_idx]
        key3 = self.key3[f_idx]
        
        start = self.batch_size * f_batch_i
        stop = self.batch_size * (f_batch_i+1)




        if(mask is None or  mask.shape[0] == 0):
            if(self.key3 is None):
                return (f[key1][start:stop], f[key2][start:stop])
            else:

                return (f[key1][start:stop], f[key2][start:stop], f[key3][start:stop])
        else:
            mask_local = mask[start:stop]
            if(key3 is None):
                return (f[key1][start:stop][mask_local], f[key2][start:stop][mask_local])
            else:

                return (f[key1][start:stop][mask_local], f[key2][start:stop][mask_local], f[key3][start:stop][mask_local])





class DataReader:
    DR_count = 0

    def __init__(self, iterable = (), **kwargs):

            #fin = None, sig_idx =1, keys = None, sig_frac = -1., data_start = 0, data_stop = -1, batch_start = -1, batch_stop = -1, batch_list = None, 
            #keep_mlow = -1., keep_mhigh = -1., hadronic_only = False, d_eta = -1., ptsort =False, randsort = False, local_storage = False, 
            #mjj_sig = -1, sig_per_batch = -1, BB_seed = 12345):

        self.ready = False
        #second  arg is default argument
        self.fin = kwargs.get('fin', None)
        self.d_eta = kwargs.get('d_eta', -1.)
        self.sig_idx = kwargs.get('sig_idx', 1)
        self.sig_frac = kwargs.get('sig_frac', -1.)
        self.sig_per_batch = kwargs.get('sig_per_batch', -1)
        self.data_start = kwargs.get('data_start', 0)
        self.data_stop = kwargs.get('data_stop', -1)
        self.hadronic_only = kwargs.get('hadronic_only', False)
        self.keep_mlow = kwargs.get('keep_mlow', -1.)
        self.keep_mhigh = kwargs.get('keep_mhigh', -1.)
        self.mjj_sig = kwargs.get('mjj_sig', -1.)
        self.BB_seed = kwargs.get('BB_seed', 123456)
        self.ptsort = kwargs.get('ptsort', False)
        self.randsort = kwargs.get('randsort', False)
        self.batch_list = kwargs.get('batch_list', None)

        local_storage = kwargs.get('local_storage', False)
        batch_start = kwargs.get('batch_start', -1)
        batch_stop = kwargs.get('batch_stop', -1)




        if(kwargs['keys'] is None):
            self.keys = ['j1_images', 'j2_images', 'mjj']
        else:
            self.keys = copy.deepcopy(kwargs.get('keys'))

        print(self.keys)

        if(self.randsort): print("Rand sort")

        self.multi_batch = False
        if(batch_start != -1 and batch_stop != -1 and self.batch_list is None):
            self.batch_list = list(range(batch_start, batch_stop+1))

        print("Batch_list:", self.batch_list)

        if(self.batch_list is not None):
            self.multi_batch = True


        self.chunk_size = 200000
        self.max_load = 1000000 #max size to load without chunks


        self.swapped_js = False
        self.first_write = True
        gpu_storage = "/storage/local/data1/gpuscratch/oamram/" 
        self.storage_dir = "" 
        if(not local_storage and os.path.isdir(gpu_storage)):
            self.storage_dir = gpu_storage

        #increment count so not to overwrite other files
        self.f_storage_name = self.storage_dir + "DReader%i_temp.h5" % DataReader.DR_count
        while(os.path.exists(self.f_storage_name)):
            DataReader.DR_count += 1
            self.f_storage_name = self.storage_dir + "DReader%i_temp.h5" % DataReader.DR_count

        print("Making temp file at %s \n" % self.f_storage_name)
        self.f_storage = h5py.File(self.f_storage_name, "w")

        self.mask = np.array([])
        #print(self.__dict__)

    def __copy__(self):
        new = DataReader()
        for attr, value in self.__dict__.items():
            new.__dict__[attr] = value
        return new

    def __deepcopy__(self, memo):
        new = DataReader()

        for attr, value in self.__dict__.items():
            new.__dict__[attr] = value

        new.keys = copy.deepcopy(self.keys)
        new.mask = np.copy(self.mask)


        return new

    def read(self):
        self.loading_images = False
        for key in self.keys:
            if('image' in key): self.loading_images = True

        
        self.nEvents = 0
        self.nTrain = 0
        self.nVal = 0

        if(self.multi_batch):
            for i in self.batch_list:
                f_name = self.fin + "BB_images_batch%i.h5" % i 
                self.read_batch(f_name)
        else:
            self.read_batch(self.fin)

        self.keys.append('label')
        print("Kept %i events after selection" % self.nEvents)
        self.ready = True

        if('swapped_js' not in self.keys and self.swapped_js): self.keys.append('swapped_js')

    def get_key(self,f, cstart, cstop, mask, key):
        if(key == 'mjj'):
            data = f["jet_kinematics"][cstart:cstop][mask,0]

        elif(key == 'j1_features'):
            j1_m = np.expand_dims(f['jet_kinematics'][cstart:cstop][mask,5], axis=-1)
            j1_feats = f['jet1_extraInfo'][cstart:cstop][mask]
            data = np.append(j1_m, j1_feats, axis = 1)
            
        elif(key == 'j2_features'):
            j2_m = np.expand_dims(f['jet_kinematics'][cstart:cstop][mask,9], axis=-1)
            j2_feats = f['jet2_extraInfo'][cstart:cstop][mask]
            data = np.append(j2_m, j2_feats, axis = 1)

        elif(key == 'jj_features'):
            j1_m = np.expand_dims(f['jet_kinematics'][cstart:cstop][mask,5], axis=-1)
            j1_feats = f['jet1_extraInfo'][cstart:cstop][mask]
            j2_m = np.expand_dims(f['jet_kinematics'][cstart:cstop][mask,9], axis=-1)
            j2_feats = f['jet2_extraInfo'][cstart:cstop][mask]
            data = np.concatenate((j1_m, j1_feats, j2_m, j2_feats), axis = 1)


        elif(key == 'j1_images' or key == 'j2_images'):
            data = np.expand_dims(f[key][cstart:cstop][mask], axis = -1)

        elif(key == 'jj_images'):
            j1_img = np.expand_dims(f['j1_images'][cstart:cstop][mask], axis = -1)
            j2_img = np.expand_dims(f['j2_images'][cstart:cstop][mask], axis = -1)
            data = np.append(j2_img, j1_img, axis = 3)

        else:
            data = f[key][cstart:cstop][mask]

        return data





    def read_batch(self, f_name):

        f = h5py.File(f_name, "r")
        if(self.data_stop == -1): stop = f['event_info'].shape[0]
        else: stop = self.stop
        self.nEvents_file = stop - self.data_start

        nChunks = 1
        if(self.nEvents_file > self.max_load and loading_images):
            nChunks = int(math.ceil(float(self.nEvents_file)/self.chunk_size))

        print("\nLoading file %s" % f_name)
        print("Will read %i events in %i chunks \n" % (self.nEvents_file, nChunks))



        for i in range(nChunks):
            cstart = self.data_start + i*self.chunk_size
            if(nChunks == 1): cstop = stop
            else: cstop = min(cstart + self.chunk_size, self.data_stop)

            #fill this chunk
            raw_labels = f['truth_label'][cstart: cstop]
            labels = np.zeros_like(raw_labels)
            if(self.sig_idx > 0):
                labels[raw_labels == self.sig_idx] = 1
                mask = np.squeeze((raw_labels <= 0) | (raw_labels == self.sig_idx)) 
            else:
                mask = np.squeeze(raw_labels >= -999999)

            #only keep some events
            cur_sig_overall = np.mean(labels[mask])
            mjj = f["jet_kinematics"][cstart:cstop,0]
            do_filter = False
            if(self.sig_per_batch >= 0):
                num_sig = np.sum(labels[mask])
                do_filter = num_sig > self.sig_per_batch
                new_sig_frac = self.sig_per_batch / labels[mask].shape[0]
                print("Signal fraction overall is %.4f, %i sig events in batch and we want %i. Filter %i" %(cur_sig_overall, num_sig, self.sig_per_batch, do_filter))
            elif(self.sig_frac >= 0.): #filter signal based on S/B in signal region
                sig_mask = (raw_labels == self.sig_idx).reshape(-1)
                bkg_mask = (raw_labels <=0).reshape(-1)
                if(self.mjj_sig < 0):
                    sig_mass = np.mean(mjj[sig_mask])
                else:
                    sig_mass = self.mjj_sig

                window_low = 0.9 * sig_mass
                window_high = 1.1* sig_mass
                in_window = (mjj > window_low) & (mjj < window_high)
                S_window = float(mjj[in_window & sig_mask].shape[0])
                B_window = float(mjj[in_window & bkg_mask].shape[0])
                cur_sig_frac_window = S_window / B_window
                do_filter = cur_sig_frac_window > self.sig_frac
                new_sig_frac = self.sig_frac/cur_sig_frac_window * cur_sig_overall
                print("Signal fraction overall is %.4f, %.4f in SR and we want %.4f in window: Filter %i" %(cur_sig_overall, cur_sig_frac_window, self.sig_frac, do_filter))

            if(do_filter): 
                #mask_sig = get_signal_mask_rand(labels, mask, new_sig_frac, self.BB_seed)
                mask_sig = get_signal_mask(labels, mask, new_sig_frac, self.BB_seed)
                mask = mask & mask_sig
            if(self.keep_mlow > 0. and self.keep_mhigh >0.):
                mjj = f["jet_kinematics"][cstart:cstop,0]
                mjj_mask = (mjj > self.keep_mlow) & (mjj < self.keep_mhigh)
                #print("mjj_mask", np.mean(mjj_mask))
                mask = mask & mjj_mask
            if(self.hadronic_only):
                is_lep = f['event_info'][cstart:cstop:,4] # stored as a float
                sig_mask = (raw_labels == self.sig_idx).reshape(-1)
                sig_mean_val = np.mean(is_lep[sig_mask] < 0.1)
                print("Hadronic only mask has mean %.3f " % sig_mean_val)
                mask = mask & (is_lep < 0.1)
            if(self.d_eta > 0.):
                deta = f['jet_kinematics'][cstart:cstop,1]
                deta_mask = deta < self.d_eta
                mask = mask & deta_mask
                #print("deta_mask", np.mean(deta_mask))



            #save labels 
            d_labels = labels[mask]
            self.nEvents += d_labels.shape[0]
            t_labels = d_labels

            if(self.first_write):
                self.f_storage.create_dataset('label', data = t_labels, chunks = True, maxshape = expandable_shape(d_labels.shape))
            else:
                append_h5(self.f_storage, 'label', t_labels)



            #save data in other keys
            data = None
            swapping_idxs = np.array([])
            if(self.ptsort):
                j1_pt = f['jet_kinematics'][cstart:cstop][mask,2]
                j2_pt = f['jet_kinematics'][cstart:cstop][mask,6]
                swapping_idxs = j2_pt > j1_pt
            elif(self.randsort):
                swapping_idxs = np.random.choice(a=[True,False], size = f['jet_kinematics'][cstart:cstop][mask].shape[0])

            for ikey,key in enumerate(self.keys):

                data = self.get_key(f,cstart, cstop, mask, key)

                if(('j1' in key or 'j2' in key) and swapping_idxs.shape[0] != 0):
                    if('j1' in key): opp_key = 'j2' + key[2:]
                    else: opp_key = 'j1' + key[2:]
                    opp_data = self.get_key(f,cstart, cstop, mask, opp_key)
                    data[swapping_idxs] = opp_data[swapping_idxs]

                #copy this chunk into class data
                tdata = None
                vdata = None
                tdata = data
                if(ikey == 0):
                    self.nTrain += tdata.shape[0]
                if(self.first_write):
                    c_shape = expandable_shape(data.shape)
                    self.f_storage.create_dataset(key, data = tdata, chunks = True, maxshape = c_shape)
                else:
                    append_h5(self.f_storage, key, tdata)


            if(swapping_idxs.shape[0] != 0):
                self.swapped_js = True
                if(self.first_write):
                    c_shape = expandable_shape(swapping_idxs.shape)
                    self.f_storage.create_dataset("swapped_js", data =swapping_idxs, chunks = True, maxshape = c_shape)
                else:
                    append_h5(self.f_storage, "swapped_js", swapping_idxs)

            self.first_write = False


        f.close()

    def make_Y_mjj(self, mjj_low, mjj_high):
        self.keys.append('Y_mjj')
        mjj = self.f_storage['mjj'][()]
        mjj_window = ((mjj > mjj_low) & (mjj < mjj_high))
        self.f_storage.create_dataset('Y_mjj', data = mjj_window)


        self.keys.append('weight')
        n_bkg_high = np.sum(mjj > mjj_high)
        n_bkg_low = np.sum(mjj < mjj_low)
        n_sig = np.sum(mjj_window)
        
        #reweight everything to have same weight as low mass bkg
        bkg_high_weight = n_bkg_low/n_bkg_high
        sig_weight = 2.*n_bkg_low / n_sig

        weights = np.ones_like(mjj, dtype=np.float32)
        weights[mjj > mjj_high] = bkg_high_weight
        weights[mjj_window] = sig_weight
        self.f_storage.create_dataset('weight', data=weights)


        del mjj, mjj_window, weights


    def make_Y_TNT(self, sig_region_cut = 0.9, bkg_region_cut = 0.2, cut_var = None, mjj_low = -999999., mjj_high = 9999999., sig_high = True, extra_str = ''):

        if(cut_var is None):
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




        keep_mask = sig_cut | bkg_cut

        Y_TNT = np.zeros_like(cut_var)

        Y_TNT[bkg_cut] = 0
        Y_TNT[sig_cut] = 1
        self.f_storage.create_dataset('Y_TNT' + extra_str, data = Y_TNT)
        self.keys.append("Y_TNT" + extra_str)

        n_bkg_high = np.sum(mjj[bkg_cut] > mjj_high)
        n_bkg_low = np.sum(mjj[bkg_cut] < mjj_low)
        n_sig = np.sum(sig_cut)
        #reweight everything to have same weight as low mass bkg
        bkg_high_weight = n_bkg_low/n_bkg_high
        sig_weight = 2.*n_bkg_low / n_sig


        self.keys.append('weight' + extra_str)
        weights = np.ones_like(mjj, dtype=np.float32)
        weights[sig_cut] = sig_weight
        weights[bkg_cut & (mjj > mjj_high)] = bkg_high_weight

        self.f_storage.create_dataset('weight' + extra_str, data=weights)

        self.apply_mask(keep_mask)






    def make_ptrw(self, Y_key, use_weights = True, normalize = True, save_plots = False, plot_dir = "", extra_str = ""):
       

        sig_cut = (self.f_storage[Y_key][()] > 0.9)
        bkg_cut = (self.f_storage[Y_key][()] < 0.1)

        if(self.mask.shape[0] != 0):
            sig_cut = sig_cut & self.mask
            bkg_cut = bkg_cut & self.mask

        if(use_weights):
            sig_weights = self.f_storage['weight'+extra_str][sig_cut]
            bkg_weights = self.f_storage['weight'+extra_str][bkg_cut]
            weights = [sig_weights, bkg_weights]
        else:
            weights = None

        print("Doing reweighting based on jet pt")
        j1_pts = self.f_storage['jet_kinematics'][:,2]
        j2_pts = self.f_storage['jet_kinematics'][:,6]

        if(self.swapped_js):
            #meaning of j1 and j2 swapped for some events
            swapped_js = self.f_storage['swapped_js'][()]
            j1_pt_temp = np.copy(j1_pts)
            j1_pts[swapped_js] = j2_pts[swapped_js]
            j2_pts[swapped_js] = j1_pt_temp[swapped_js]
            del j1_pt_temp


        j1_sr_pts = j1_pts[sig_cut]
        j1_br_pts = j1_pts[bkg_cut]

        j2_sr_pts = j2_pts[sig_cut]
        j2_br_pts = j2_pts[bkg_cut]

        labels = ['Signal Region', 'Background Region']
        colors = ['b', 'r']
        n_pt_bins = 20

        j1_bins, j1_ratio = make_ratio_histogram([j1_sr_pts, j1_br_pts], labels, colors, 'jet1 pt (GeV)', "Jet1 Sig vs. Bkg Pt distribution", n_pt_bins,
                        normalize=normalize, weights = weights, save = save_plots, fname=plot_dir + "j1_ptrw.png")
        j1_rw_idxs = np.digitize(j1_pts, bins = j1_bins) - 1
        
        j1_rw_idxs = np.clip(j1_rw_idxs, 0, len(j1_ratio)-1) #handle overflows
        j1_rw_vals = j1_ratio[j1_rw_idxs]
        #don't reweight signal region
        j1_rw_vals[sig_cut] = 1.
        if(use_weights):
            j1_rw_vals *= self.f_storage['weight' + extra_str]

        self.f_storage.create_dataset('j1_ptrw'+extra_str, data = j1_rw_vals)
        self.keys.append("j1_ptrw"+extra_str)

        j2_bins, j2_ratio = make_ratio_histogram([j2_sr_pts, j2_br_pts], labels, colors, 'jet2 pt (GeV)', "Jet2 Sig vs. Bkg Pt distribution", n_pt_bins,
                        normalize=normalize, weights = weights, save = save_plots, fname=plot_dir + "j2_ptrw.png")
        j2_rw_idxs = np.digitize(j2_pts, bins = j2_bins) - 1
        
        j2_rw_idxs = np.clip(j2_rw_idxs, 0, len(j2_ratio)-1) #handle overflows
        j2_rw_vals = j2_ratio[j2_rw_idxs]
        #don't reweight signal region
        j2_rw_vals[sig_cut] = 1.

        if(use_weights):
            j2_rw_vals *= self.f_storage['weight'+extra_str]


        self.f_storage.create_dataset('j2_ptrw'+extra_str, data = j2_rw_vals)
        self.keys.append("j2_ptrw" + extra_str)



    def __getitem__(self, key):
        if(not self.ready):
            print("Datareader has not loaded data yet! Must call read() first ")
            exit(1)
        if(key not in self.keys):
            print("Key %s not in list of preloaded keys!" % key, self.keys)
            exit(1)
        mask_ = self.mask

        if(mask_.shape[0] == 0):
            return self.f_storage[key][()]
        else:
            return self.f_storage[key][()][mask_]

    def gen(self, key1, key2, key3 = None, batch_size = 256):
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

        mask_ = self.mask


        h5_gen = MyGenerator(self.f_storage, n_objs, batch_size, key1, key2, key3, mask = mask_)
        return h5_gen

    def labeler_scores(self, model, key, chunk_size = 10000):
        
        n_objs = self.f_storage[key].shape[0]
        n_chunks = int(np.ceil(n_objs / chunk_size))
        results = np.array([])
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
            print("Deprecated option")
            sys.exit(1)

    

    def cleanup(self):
        print("Cleaning up temp file %s" % self.f_storage_name)
        os.system("rm %s" % self.f_storage_name)


    def __del__(self):
        self.cleanup()

    
        

    



#create a mask that removes signal events to enforce a given fraction
#randomly selects which signal events to remove
def get_signal_mask(labels, mask, sig_frac, BB_seed=12345):

    np.random.seed(BB_seed)
    num_events = labels.shape[0]
    cur_frac =  np.mean(labels[mask])
    keep_frac = (sig_frac/cur_frac)
    drop_frac = (1. - (sig_frac/cur_frac))
    progs = np.cumsum(labels)/(num_events * cur_frac)
    all_idxs = np.arange(num_events)
    sig_idxs = all_idxs[labels.reshape(-1) == 1]
    num_drop = int(sig_idxs.shape[0]*drop_frac)
    drop_sigs = np.random.choice(sig_idxs, num_drop, replace = False)
    keep_idxs = np.array([True]*num_events)
    keep_idxs[drop_sigs] = False
    return keep_idxs


#create a mask that removes signal events to enforce a given fraction
#Random chance to keep each signal event
def get_signal_mask_rand(labels, mask, sig_frac, BB_seed=12345):

    np.random.seed(BB_seed)
    num_events = labels.shape[0]
    cur_frac =  np.mean(labels[mask])
    if(cur_frac <= sig_frac):
        return np.ones_like(labels)

    drop_frac = (1. - (sig_frac/cur_frac))
    rands = np.random.random(num_events)
    keep_idxs = (labels.reshape(-1) == 0) | (rands > drop_frac)
    return keep_idxs
