import h5py
import numpy as np
import math
import tensorflow as tf 
import copy
import os
import sys
from .PlotUtils import *
from .Consts import *
from sklearn.preprocessing import StandardScaler



def expandable_shape(d_shape):
    c_shape = list(d_shape)
    c_shape[0] = None
    c_shape = tuple(c_shape)
    return c_shape


def append_h5(f, name, data):
    prev_size = f[name].shape[0]
    f[name].resize(( prev_size + data.shape[0]), axis=0)
    f[name][prev_size:] = data


def mjj_from_4vecs(j1, j2):
    #assume j1 and j2 in (pt,eta,phi,m) format
    px1 = j1[:,0] * np.cos(j1[:,2])  
    py1 = j1[:,0] * np.sin(j1[:,2])  
    pz1 = j1[:,0] * np.sinh(j1[:,1]) 
    px2 = j2[:,0] * np.cos(j2[:,2])  
    py2 = j2[:,0] * np.sin(j2[:,2])  
    pz2 = j2[:,0] * np.sinh(j2[:,1]) 
    E1 = np.sqrt(px1**2 + py1**2 + pz1**2 + j1[:,3]**2)
    E2 = np.sqrt(px2**2 + py2**2 + pz2**2 + j2[:,3]**2)
    mjj = np.sqrt((E1 + E2)**2 - (px1 + px2)**2 - (py1 + py2)**2 - (pz1 + pz2)**2)
    return mjj

def get_avg_sys_reweight(f, sys, deta, deta_min, lund_weights = True):
    if('btag' in sys): #shape only, don't change normalization
        return 1.0
    deta_min = max(0., deta_min)
    deta = min(10., deta)
    deta_var = f['jet_kinematics'][:,1]
    deta_mask = (deta_var < deta) & (deta_var > deta_min)
    nom_weight = f['sys_weights'][:,0][deta_mask]
    if(lund_weights): nom_weight *= f['lund_weights'][:,0][deta_mask]

    if('lund' in sys):
        weight_idx = lund_weights_map[sys]
        sig_weights = nom_weight * f['lund_weights_sys_var'][:,weight_idx][deta_mask]

    else:
        weight_idx = sys_weights_map[sys]
        sig_weights = nom_weight * f['sys_weights'][:,weight_idx][deta_mask]

    return np.mean(sig_weights) / np.mean(nom_weight)

def get_sideband_mask(f):
    deta = f['jet_kinematics'][:,1]
    mjj = f['jet_kinematics'][:,0]
    pt1= f['jet_kinematics'][:,2]
    pt2 = f['jet_kinematics'][:,6]
    pt3 = f['jet_kinematics'][:,10]
    deta_mask = (deta > 2.0 ) & (deta < 2.5)
    jet3_mask = pt3 < 300.
    #2.0<dEta<2.5 && ( |(pt1-pt2)/(p1+pt2)|>0.1 || 2*pt1*pt2*(coshdEta+1)/mjj^2>1 ||  2*pt1*pt2*(coshdEta+1)/mjj^2<0.95 ) && pt3<300 GeV
    pt_asym_mask = ((np.abs(pt1 - pt2) / (pt1 + pt2)) > 0.1) | (2. * pt1 * pt2 * np.cosh(deta + 1) / mjj**2 > 1.) | (2. * pt1 * pt2 * np.cosh(deta + 1) / mjj**2 < 0.95) 
    return deta_mask & jet3_mask & pt_asym_mask

def apply_norm(d, norm):
    if(norm is None): return d
    else: return (d - norm[0]) / norm[1]



class MyGenerator(tf.keras.utils.Sequence):
    def __init__(self, f, n, batch_size, key1, key2, key3 = None, mask = None, key1_norm = None, key2_norm = None, key3_norm = None):
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


        self.key1_norm = key1_norm
        self.key2_norm = key2_norm
        self.key3_norm = key3_norm


    def add_dataset(self, key1,key2, key3 = None, f2 = None, n2 = None, mask = None, dataset = None):
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
            if(key3 is None):
                return (apply_norm(f[key1][start:stop], self.key1_norm), apply_norm(f[key2][start:stop], self.key2_norm))
            else:
                return (apply_norm(f[key1][start:stop], self.key1_norm), apply_norm(f[key2][start:stop], self.key2_norm), 
                        apply_norm(f[key3][start:stop], self.key3_norm))
        else:
            mask_local = mask[start:stop]
            if(key3 is None):
                return (apply_norm(f[key1][start:stop][mask_local], self.key1_norm), apply_norm(f[key2][start:stop][mask_local], self.key2_norm))
            else:

                return (apply_norm(f[key1][start:stop][mask_local], self.key1_norm), apply_norm(f[key2][start:stop][mask_local], self.key2_norm), 
                        apply_norm(f[key3][start:stop][mask_local], self.key3_norm))





class DataReader:
    DR_count = 0

    def __init__(self, iterable = (), **kwargs):

        self.ready = False
        #second  arg is default argument
        self.fin = kwargs.get('fin', None)
        self.deta = kwargs.get('deta', -1.)
        self.deta_min = kwargs.get('deta_min', -1.)
        self.data_start = kwargs.get('data_start', 0)
        self.data_stop = kwargs.get('data_stop', -1)
        self.hadronic_only = kwargs.get('hadronic_only', False)
        self.sideband = kwargs.get('sideband', False)
        self.keep_mlow = kwargs.get('keep_mlow', -1.)
        self.keep_mhigh = kwargs.get('keep_mhigh', -1.)
        self.mjj_sig = kwargs.get('mjj_sig', -1.)
        self.save_mem = kwargs.get('save_mem', False)

        self.data = kwargs.get('data', False)

        self.max_events = kwargs.get('max_events', -1)

        self.keep_tau1  = kwargs.get('keep_tau1', False)



        self.sig_idx = kwargs.get('sig_idx', 1)
        self.sig_file = kwargs.get('sig_file', '')
        self.sig_weights = kwargs.get('sig_weights', False)
        self.lund_weights = kwargs.get('lund_weights', True)
        self.sig_frac = kwargs.get('sig_frac', -1.)
        self.sig_per_batch = kwargs.get('sig_per_batch', -1.0)
        self.spb_before_selection = kwargs.get('spb_before_selection', False)

        self.sig_sys = kwargs.get('sig_sys', '')
        self.sig_only = kwargs.get('sig_only', False)

        if (len(self.sig_sys) > 0):
            if(self.sig_sys not in sys_weights_map.keys() and self.sig_sys not in JME_vars and self.sig_sys not in Lund_vars):
                print("Un recognized systematic %s! " % self.sig_sys)
                sys.exit(1)

        self.BB_seed = kwargs.get('BB_seed', 123456)
        #self.rng = np.random.Generator(np.random.PCG64(BB_seed))
        self.rng = np.random
        self.rng.seed(self.BB_seed)
        self.ptsort = kwargs.get('ptsort', False)
        self.randsort = kwargs.get('randsort', False)
        self.batch_list = kwargs.get('batch_list', None)
        self.no_minor_bkgs = kwargs.get('no_minor_bkgs', False)
        self.replace_ttbar = kwargs.get('replace_ttbar', False)


        local_storage = kwargs.get('local_storage', False)
        batch_start = kwargs.get('batch_start', -1)
        batch_stop = kwargs.get('batch_stop', -1)

        self.keep_LSF = kwargs.get('keep_LSF', True)
        self.clip_feats = kwargs.get('clip_feats', True)
        self.nsubj_ratios = kwargs.get('nsubj_ratios', True)
        self.clip_pts = kwargs.get('clip_pts', True)

        self.verbose = kwargs.get('verbose', True)


        if(self.verbose): print("Creating dataset. Mass range %.0f - %.0f. Delta Eta %.1f -  %.1f" % (self.keep_mlow, self.keep_mhigh, self.deta_min, self.deta))

        self.sys_norm_reweight = 1.0
        if(len(self.sig_file ) > 0 ):
            if(self.verbose): print("Loading signal %s " % self.sig_file)
            self.sig_file_h5 = h5py.File(self.sig_file, "r")

            if(len(self.sig_sys) > 0 and self.sig_sys in (set(sys_weights_map.keys()) | Lund_vars)  ):
                #change normalization of signal
                self.sys_norm_reweight = get_avg_sys_reweight(self.sig_file_h5, self.sig_sys, self.deta, self.deta_min, lund_weights = self.lund_weights)
                if(self.verbose): print("Sig norm reweight is %.4f" % self.sys_norm_reweight)

        self.sep_signal = len(self.sig_file) > 0 and not self.sig_only




        keys = kwargs.get('keys', None)
        if(keys is None):
            self.keys = {'j1_images', 'j2_images', 'mjj'}
        else:
            self.keys = set(copy.deepcopy(kwargs.get('keys')))

            if(self.verbose): print(self.keys)

        if(self.randsort and self.verbose): print("Rand sort")

        self.multi_batch = False
        if(batch_start != -1 and batch_stop != -1 and self.batch_list is None):
            self.batch_list = list(range(batch_start, batch_stop+1))

        if(self.verbose): print("Batch_list:", self.batch_list)

        if(self.batch_list is not None):
            self.multi_batch = True


        self.chunk_size = 1000000
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

        if(self.verbose): print("Making temp file at %s \n" % self.f_storage_name)
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

    def process_feats(self, feats):
        #Features are tau1, tau2, tau3, tau4, LSF, DeepB, nPF
        nonLSF_idxs = [0,1,2,3,5,6]
        b_idx = -2
        lsf_idx = 4
        rfeats = np.copy(feats)
        if(self.keep_LSF): 
            if(self.clip_feats):
                rfeats[:,lsf_idx] = np.clip(rfeats[:,lsf_idx], 0., 1.)

        else: 
            rfeats = rfeats [:,nonLSF_idxs]


        if(self.clip_feats):
            rfeats[:,b_idx] = np.clip(rfeats[:,b_idx], 0., None)
            #rfeats[:, 0] = np.clip(rfeats[:,0], 0., 0.45)

        if(self.nsubj_ratios):
            #Make tau2 tau3 and tau4  variables tau21, tau32, tau43 respectively
            eps = 1e-6
            rfeats[:,1] = feats[:,1] / (feats[:,0] + eps)
            rfeats[:,2] = feats[:,2] / (feats[:,1] + eps)
            rfeats[:,3] = feats[:,3] / (feats[:,2] + eps)

        #don't use tau1
        if(not self.keep_tau1):
            rfeats = rfeats[:,1:]

        return rfeats


    def read(self):
        self.loading_images = False
        for key in self.keys:
            if('image' in key): self.loading_images = True

        
        self.nEvents = 0
        self.nTrain = 0
        self.nVal = 0


        if(self.data):
            self.num_total_batches = 40
        else:
            self.num_total_batches = 80
        self.sig_marker = 0
        self.sig_chunk_size  = 1
        self.batchIdx = -1

        if(len(self.sig_file) > 0):
            self.nsig_in_file = self.sig_file_h5['event_info'].shape[0]

            #self.sig_chunk_size = int(self.nsig_in_file / (2. * len(self.batch_list)))
            self.sig_chunk_size = int(self.nsig_in_file / self.num_total_batches)
            
            if(self.sig_chunk_size <= self.sig_per_batch):
                print("WARNING: Not enough signal to split across 40 batches, splitting only across this sample, could cause repeats in later runs!")
                self.sig_chunk_size = int(self.nsig_in_file / (len(self.batch_list)))


        if(self.sig_only):
            self.data_start = np.amin(self.batch_list) * self.sig_chunk_size
            self.data_stop = self.data_start + self.sig_chunk_size * len(self.batch_list)
            self.read_batch(self.sig_file)

        elif(self.multi_batch):
            for i in self.batch_list:
                if(self.max_events > 0 and self.nEvents >= self.max_events): 
                    print("%i events read, above max (%i). Stopping loading" % (self.nEvents, self.max_events))
                    break
                self.batchIdx = i
                self.sig_marker = self.sig_chunk_size * i
                if(self.data): preface = 'data_'
                else: preface = 'BB_'
                f_name = self.fin + preface + "images_batch%i.h5" % i 
                if(not os.path.exists(f_name)):
                    f_name = self.fin + preface + "batch%i.h5" % i
                self.read_batch(f_name)
                if(self.save_mem):
                    os.system("rm " + f_name)
        else:
            self.read_batch(self.fin)

        self.keys.add('label')
        print("Kept %i events after selection" % self.nEvents)
        self.ready = True

        if('swapped_js' not in self.keys and self.swapped_js): self.keys.add('swapped_js')


    def get_key(self,f, cstart, cstop, mask, key):
        if(key == 'mjj'):
            data = f["jet_kinematics"][cstart:cstop][mask,0]

        elif(key == 'j1_features'):
            j1_m = np.expand_dims(f['jet_kinematics'][cstart:cstop][mask,5], axis=-1)
            j1_feats = self.process_feats(f['jet1_extraInfo'][cstart:cstop][mask])
            data = np.append(j1_m, j1_feats, axis = 1)
            
        elif(key == 'j2_features'):
            j2_m = np.expand_dims(f['jet_kinematics'][cstart:cstop][mask,9], axis=-1)
            j2_feats = self.process_feats(f['jet2_extraInfo'][cstart:cstop][mask])
            data = np.append(j2_m, j2_feats, axis = 1)


        elif(key == 'jj_features'):
            j1_m = np.expand_dims(f['jet_kinematics'][cstart:cstop][mask,5], axis=-1)
            j1_feats = self.process_feats(f['jet1_extraInfo'][cstart:cstop][mask])
            j2_m = np.expand_dims(f['jet_kinematics'][cstart:cstop][mask,9], axis=-1)
            j2_feats = self.process_feats(f['jet2_extraInfo'][cstart:cstop][mask])
            data = np.concatenate((j1_m, j1_feats, j2_m, j2_feats), axis = 1)


        elif(key == 'j1_images' or key == 'j2_images'):
            data = np.expand_dims(f[key][cstart:cstop][mask], axis = -1)

        elif(key == 'jj_images'):
            j1_img = np.expand_dims(f['j1_images'][cstart:cstop][mask], axis = -1)
            j2_img = np.expand_dims(f['j2_images'][cstart:cstop][mask], axis = -1)
            data = np.append(j2_img, j1_img, axis = 3)

        #fix naming issue (j1 vs jet1)
        elif(key == 'j1_JME_vars'): data = f['jet1_JME_vars'][cstart:cstop][mask]
        elif(key == 'j2_JME_vars'): data = f['jet2_JME_vars'][cstart:cstop][mask]

        else:
            data = f[key][cstart:cstop][mask]

        return data





    def read_batch(self, f_name):

        f = h5py.File(f_name, "r")
        if(self.data_stop == -1): stop = f['event_info'].shape[0]
        else: stop = self.data_stop
        self.nEvents_file = stop - self.data_start

        nChunks = 1
        if(self.nEvents_file > self.max_load and self.loading_images):
            nChunks = int(math.ceil(float(self.nEvents_file)/self.chunk_size))
            if(self.sig_per_batch > 0):
                self.sig_per_batch /= nChunks
            

        if(self.verbose): print("\nLoading file %s" % f_name)
        if(self.verbose): print("Will read %i events in %i chunks \n" % (self.nEvents_file, nChunks))



        for i in range(nChunks):
            cstart = self.data_start + i*self.chunk_size
            if(nChunks == 1): cstop = stop
            else: cstop = min(cstart + self.chunk_size, self.data_stop)

            #fill this chunk
            raw_labels = f['truth_label'][cstart: cstop]
            #QCD 0, single top -1, ttbar -2, V+jets -3
            labels = np.copy(raw_labels)
            if((not self.sep_signal) and self.sig_idx > 0):
                labels[raw_labels == self.sig_idx] = 1
                if(self.no_minor_bkgs): 
                    mask = np.squeeze((raw_labels == 0) | (raw_labels == self.sig_idx)) 
                else:
                    mask = np.squeeze((raw_labels <= 0) | (raw_labels == self.sig_idx)) 

            elif(self.sep_signal):
                if(self.no_minor_bkgs): 
                    mask = np.squeeze((raw_labels == 0))
                else:
                    mask = np.squeeze((raw_labels <= 0))
            else:
                mask = np.squeeze(raw_labels >= -999999)

            num_ttbar = 0
            if(self.replace_ttbar):
                num_ttbar = np.sum(raw_labels == -2)
                mask = mask & np.squeeze(raw_labels != -2)
                print("Removing %i ttbar events" % num_ttbar)

            #Selection masks before reducing signal
            if(self.hadronic_only):
                is_lep = f['event_info'][cstart:cstop:,4] # stored as a float
                mask = mask & (is_lep < 0.1)
                if(not self.sep_signal):
                    sig_mask = (raw_labels == self.sig_idx).reshape(-1)
                    sig_mean_val = np.mean(is_lep[sig_mask] < 0.1)
                    if(self.verbose): print("Hadronic only mask has mean %.3f " % sig_mean_val)
            if(self.keep_mlow > 0. and self.keep_mhigh >0.):
                mjj = f["jet_kinematics"][cstart:cstop,0]
                mjj_mask = (mjj > self.keep_mlow) & (mjj < self.keep_mhigh)
                #print("mjj_mask", np.mean(mjj_mask))
                mask = mask & mjj_mask
            if(self.sideband):
                sb_mask = get_sideband_mask(f)
                if(self.verbose): print("SB mask eff", np.mean(sb_mask))
                mask = mask & get_sideband_mask(f)
            elif(self.deta > 0. or self.deta_min > 0.):
                deta = f['jet_kinematics'][cstart:cstop,1]
                if(self.deta > 0.): mask = mask & (deta < self.deta)
                if(self.deta_min > 0.): mask = mask & (deta > self.deta_min)


            #only keep some events
            mjj = f["jet_kinematics"][cstart:cstop,0]
            do_filter = False
            sig_to_keep = -1
            if(not self.sep_signal):
                if(self.sig_per_batch >= 0):
                    num_sig = np.sum(labels[mask] > 0)
                    do_filter = num_sig > self.sig_per_batch
                    sig_to_keep = self.sig_per_batch
                    if(self.verbose): print("%i sig events in batch and we want %i. Filter %i" %(num_sig, self.sig_per_batch, do_filter))
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
                    num_sig_overall = np.sum(labels[mask] > 0)
                    sig_to_keep = int(self.sig_frac/cur_sig_frac_window * num_sig_overall)
                    if(self.verbose) : print("Num sig overall is %i, Sig frac %.4f in SR and we want %.4f in window: Filter %i" %(num_sig_overall, cur_sig_frac_window, self.sig_frac, do_filter))
                if(do_filter): 
                    #mask_sig = get_signal_mask_rand(labels, mask, new_sig_frac, self.BB_seed)
                    mask_sig = get_signal_mask(labels, mask, sig_to_keep, self.rng)
                    mask = mask & mask_sig



            #save labels 
            t_labels = labels[mask]


            #save data in other keys
            data = None
            swapping_idxs = np.array([])
            if(self.ptsort):
                j1_pt = f['jet_kinematics'][cstart:cstop][mask,2]
                j2_pt = f['jet_kinematics'][cstart:cstop][mask,6]
                swapping_idxs = j2_pt > j1_pt
            elif(self.randsort):
                swapping_idxs = self.rng.choice(a=[True,False], size = f['jet_kinematics'][cstart:cstop][mask].shape[0])


            num_sig_inj = 0
            #load separate signal data
            if(self.sep_signal and self.sig_per_batch > 0):
                sig_mask_temp = np.ones(self.sig_chunk_size, dtype = bool)
                s_start = self.sig_marker
                s_stop = self.sig_marker + self.sig_chunk_size

                if(s_stop > self.nsig_in_file):
                    print("Not enough signal events in file (more than %i), exiting" % self.nsig_in_file)
                    exit(1)
                if(self.hadronic_only):
                    is_lep = self.sig_file_h5['event_info'][s_start:s_stop:,4] # stored as a float
                    sig_mask_temp = sig_mask_temp & (is_lep < 0.1)
                    if(self.verbose): print("Hadronic only mask has mean %.3f " % np.mean(is_lep < 0.1))

                #local copy to allow JME modifications
                sig_jet_kinematics = np.copy(self.sig_file_h5["jet_kinematics"][s_start:s_stop])
                if(len(self.sig_sys) > 0 and self.sig_sys in JME_vars):
                    #Apply JME variation
                    m_idx = JME_vars_map["m_" + self.sig_sys]
                    j1_m_corr = self.sig_file_h5["jet1_JME_vars"][s_start:s_stop, m_idx]
                    j2_m_corr = self.sig_file_h5["jet2_JME_vars"][s_start:s_stop, m_idx]


                    sig_jet_kinematics [:, 5] = j1_m_corr
                    sig_jet_kinematics [:, 9] = j2_m_corr

                    if('JE' in self.sig_sys): #also correct pt (JES and JER)
                        pt_idx = JME_vars_map["pt_" + self.sig_sys]
                        j1_pt_corr = self.sig_file_h5["jet1_JME_vars"][s_start:s_stop, pt_idx]
                        j2_pt_corr = self.sig_file_h5["jet2_JME_vars"][s_start:s_stop, pt_idx]

                        sig_jet_kinematics [:, 2] = j1_pt_corr
                        sig_jet_kinematics [:, 6] = j2_pt_corr


                    new_mjj = mjj_from_4vecs(sig_jet_kinematics[:, 2:6], sig_jet_kinematics[:, 6:10])
                    sig_jet_kinematics [:,0] = new_mjj

                sig_mjj_eff = sig_deta_eff = 1.
                if(self.keep_mlow > 0. and self.keep_mhigh >0.):
                    mjj = sig_jet_kinematics[:,0]
                    mjj_mask = (mjj > self.keep_mlow) & (mjj < self.keep_mhigh)
                    sig_mask_temp = sig_mask_temp & mjj_mask
                    sig_mjj_eff = np.mean(mjj_mask)
                    if(self.hadronic_only):
                        sig_mjj_eff = np.mean(mjj_mask[is_lep < 0.1]) #compute mjj eff for hadronic only

                if(self.deta > 0. or self.deta_min > 0.):
                    deta = sig_jet_kinematics[:,1]
                    deta_mask = deta > -99999
                    if(self.deta > 0.): deta_mask = deta_mask & (deta < self.deta)
                    if(self.deta_min > 0.): deta_mask = deta_mask & (deta > self.deta_min)
                    sig_mask_temp = sig_mask_temp & deta_mask
                    sig_deta_eff = np.mean(deta_mask)
                    if(self.hadronic_only):
                        sig_deta_eff = np.mean(deta_mask[is_lep < 0.1]) #compute deta eff for hadronic only

                idx_list = np.arange(0, self.sig_chunk_size)
                num_sig_inj = self.sig_per_batch
                if(self.replace_ttbar):
                    ttbar_norm_change = self.sig_file_h5['top_ptrw_avg'][0]
                    num_sig_inj = num_ttbar * ttbar_norm_change
                    self.spb_before_selection = True

                if(self.sig_weights): #do weighted random sampling of signal events
                    sig_weights = self.sig_file_h5['sys_weights'][s_start:s_stop, 0][sig_mask_temp]
                    if(self.lund_weights): sig_weights *= self.sig_file_h5['lund_weights'][s_start:s_stop][sig_mask_temp]

                    if(len(self.sig_sys) > 0 and self.sig_sys in sys_weights_map.keys()):
                        weight_idx = sys_weights_map[self.sig_sys]
                        sig_weights *= self.sig_file_h5['sys_weights'][s_start:s_stop,weight_idx][sig_mask_temp]
                        num_sig_inj *= self.sys_norm_reweight

                    elif(len(self.sig_sys) > 0 and self.sig_sys in Lund_vars):
                        weight_idx = Lund_vars_map[self.sig_sys]
                        sig_weights *= self.sig_file_h5['lund_weights_sys_var'][s_start:s_stop,weight_idx][sig_mask_temp]
                        num_sig_inj *= self.sys_norm_reweight

                    #used for the random sampling
                    sum_sig_weight = np.sum(sig_weights)
                    probs = sig_weights / sum_sig_weight


                if(self.spb_before_selection):
                    num_sig_inj *= sig_mjj_eff * sig_deta_eff
                    if(self.verbose): print("Sig mjj eff %.4f deta eff %.4f. Spb now %i" % (sig_mjj_eff, sig_deta_eff, num_sig_inj))

                #num_sig_inj = int(round(num_sig_inj))

                if( abs(num_sig_inj - np.round(num_sig_inj)) > 0.01):
                    fraction = num_sig_inj - np.floor(num_sig_inj)
                    num_batches_extra_sig = int(round(fraction * self.num_total_batches))
                    spacing = np.floor(self.num_total_batches / num_batches_extra_sig)
                    batches_extra_sig = [(i * spacing)%self.num_total_batches for i in range(num_batches_extra_sig)]
                    if(self.batchIdx in batches_extra_sig):
                        num_sig_inj = int(np.floor(num_sig_inj)) + 1
                    else:
                        num_sig_inj = int(np.floor(num_sig_inj))

                else:
                    num_sig_inj = int(round(num_sig_inj))

                if(self.verbose): print("Injecting %i signal events" % num_sig_inj)

                    

                if(idx_list[sig_mask_temp].shape[0] < num_sig_inj):
                    print("Not enough signal in chunk after selection (%i, want %i)! Exiting" % (idx_list[sig_mask_temp].shape[0], num_sig_inj))
                    sys.exit(1)


                if(self.sig_weights): sig_idxs = self.rng.choice(idx_list[sig_mask_temp], num_sig_inj, replace = False, p = probs)
                else: sig_idxs = idx_list[sig_mask_temp][:num_sig_inj]
                sig_final_mask = np.zeros(self.sig_chunk_size, dtype = bool)
                sig_final_mask[sig_idxs] = True

                if(self.ptsort):
                    j1_pt_sig = sig_jet_kinematics[sig_final_mask, 2]
                    j2_pt_sig = sig_jet_kinematics[sig_final_mask, 6]
                    swapping_idxs = np.append(swapping_idxs, j2_pt_sig > j1_pt_sig, axis=0)

                elif(self.randsort):
                    swapping_idxs = np.append(swapping_idxs, self.rng.choice(a=[True,False], size = num_sig_inj), axis=0)


            #combine labels
            if(self.sep_signal and num_sig_inj > 0): 
                extra_labels = np.expand_dims(np.ones(num_sig_inj, dtype=np.int8),axis=-1)
                if(self.replace_ttbar): np.expand_dims(np.array([-2]*num_sig_inj, dtype = np.int8), axis = -1)
                t_labels = np.append(t_labels, extra_labels, axis=0)
            #print("# sig events: ", np.sum(t_labels == 1))
            self.nEvents += t_labels.shape[0]


            #Shuffle before saving
            if(self.sep_signal and num_sig_inj > 0):
                shuffle_order = np.random.permutation(t_labels.shape[0])
                if(swapping_idxs.shape[0] > 0):
                    swapping_idxs = swapping_idxs[shuffle_order]
            else:
                shuffle_order = np.arange(t_labels.shape[0])

            #save labels
            t_labels = t_labels[shuffle_order]
            if(self.first_write):
                self.f_storage.create_dataset('label', data = t_labels, chunks = True, maxshape = expandable_shape(t_labels.shape))
            else:
                append_h5(self.f_storage, 'label', t_labels)


            for ikey,key in enumerate(self.keys):

                data = self.get_key(f,cstart, cstop, mask, key)

                if(self.sep_signal and num_sig_inj > 0): 
                    if(key == 'jet_kinematics'): sig_data = sig_jet_kinematics[sig_final_mask]
                    else: sig_data = self.get_key(self.sig_file_h5, s_start, s_stop, sig_final_mask, key)
                    data = np.append(data, sig_data, axis= 0)
                    data = data[shuffle_order]

                if(('j1' in key or 'j2' in key) and swapping_idxs.shape[0] != 0):
                    if('j1' in key): opp_key = 'j2' + key[2:]
                    else: opp_key = 'j1' + key[2:]
                    opp_data = self.get_key(f,cstart, cstop, mask, opp_key)
                    if(self.sep_signal and num_sig_inj > 0): opp_data = np.append(opp_data, self.get_key(self.sig_file_h5, s_start, s_stop, sig_final_mask, opp_key), axis=0)
                    data[swapping_idxs] = opp_data[shuffle_order][swapping_idxs]




                #copy this chunk into class data
                tdata = data
                if(ikey == 0):
                    self.nTrain += tdata.shape[0]
                if(self.first_write):
                    c_shape = expandable_shape(data.shape)
                    self.f_storage.create_dataset(key, data = tdata, chunks = True, maxshape = c_shape, compression = 'gzip')
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

    #depricated
    def standard_scaler(self, key, weights_key = None, scaler = None):
        data = self.__getitem__(key)
        weights = None
        if(scaler is None):
            scaler = StandardScaler()
            if(weights_key is not None):
                weights = self.__getitem__(weights_key)
            scaler.fit(data, sample_weight = weights)


        if(self.mask.shape[0] != 0):
            self.f_storage[key][self.mask] = scaler.transform(data)
        else:
            self.f_storage[key][()] = scaler.transform(data)
        return scaler


    def make_Y_ttbar(self, jet_features, ptcut = 400., tau32_cut = 0.54,  deepcsv_cut = 0.16, extra_str = ''):


        j_m = jet_features[:,0]
        j_tau32 = jet_features[:,3]
        j_deepcsv = jet_features[:,6]


        ptcut_mask = (self.f_storage['jet_kinematics'][:,2] > ptcut) & ( self.f_storage['jet_kinematics'][:,6] > ptcut)

        tag_selection = (j_tau32 < tau32_cut) & (j_deepcsv > deepcsv_cut) & ptcut_mask


        SR = tag_selection & (j_m > 105) & (j_m < 220)
        bkg_high = tag_selection & (j_m > 220)
        bkg_low = tag_selection & (j_m < 105)

        n_bkg_high = np.sum(bkg_high)
        n_bkg_low = np.sum(bkg_low)
        n_sig = np.sum(SR)

        #reweight everything to have same weight as low mass bkg
        bkg_high_weight = n_bkg_low/n_bkg_high
        sig_weight = 2.*n_bkg_low / n_sig

        self.keys.add('weight' + extra_str)
        weights = np.ones_like(j_m, dtype=np.float32)
        weights[bkg_high] = bkg_high_weight
        weights[SR] = sig_weight
        self.f_storage.create_dataset('weight' + extra_str, data=weights)




        Y_ttbar = np.zeros_like(j_m)
        Y_ttbar[SR] = 1

        self.keys.add('Y_ttbar' + extra_str)
        self.f_storage.create_dataset('Y_ttbar' + extra_str, data = Y_ttbar)

        filter_frac = self.apply_mask(tag_selection)
        return filter_frac


    def make_Y_mjj(self, mjj_low, mjj_high):
        self.keys.add('Y_mjj')
        mjj = self.f_storage['mjj'][()]
        mjj_window = ((mjj > mjj_low) & (mjj < mjj_high))
        self.f_storage.create_dataset('Y_mjj', data = mjj_window)


        self.keys.add('weight')
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


    def make_Y_TNT(self, sig_region_cut = 0.9, bkg_region_cut = 0.2, cut_var = None, mjj_low = -999999., mjj_high = 9999999., sig_high = True, extra_str = '',
                    bkg_cut_type = 0):

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
            mjj_sig_window = ((mjj > mjj_low) & (mjj < mjj_high))
            if(bkg_cut_type == 0): #bkg in sb's only
                mjj_bkg_window = ((mjj < mjj_low) | (mjj > mjj_high))
                bkg_cut = bkg_cut & mjj_bkg_window
            elif(bkg_cut_type == 1): #bkg in whole region
                mjj_bkg_window = mjj > 0
                bkg_cut = bkg_cut & mjj_bkg_window
            elif(bkg_cut_type == 2): #bkg in signal region only
                mjj_bkg_window = mjj_sig_window
                bkg_cut = bkg_cut & mjj_bkg_window
            elif(bkg_cut_type ==3): #bkg cut from AE OR sidebands
                mjj_bkg_window = ((mjj < mjj_low) | (mjj > mjj_high))
                bkg_cut = bkg_cut | mjj_bkg_window
            else:
                print("Invalid bkg cut type %i ? (0-3 allowed)")
                sys.exit(1)

            sig_cut = sig_cut & mjj_sig_window




        keep_mask = sig_cut | bkg_cut

        Y_TNT = np.zeros_like(cut_var)

        Y_TNT[bkg_cut] = 0
        Y_TNT[sig_cut] = 1
        self.f_storage.create_dataset('Y_TNT' + extra_str, data = Y_TNT)
        self.keys.add("Y_TNT" + extra_str)

        n_bkg_high = np.sum(mjj[bkg_cut] > mjj_high)
        n_bkg_low = np.sum(mjj[bkg_cut] < mjj_low)
        n_bkg_mid = mjj[bkg_cut & mjj_sig_window].shape[0]
        n_sig = np.sum(sig_cut)
        #reweight everything to have same weight as low mass bkg
        if(n_bkg_high > 0 and n_bkg_low > 0):
            bkg_high_weight = n_bkg_low/n_bkg_high
        else:
            bkg_high_weight = 1.

        if(n_bkg_mid > 0 and n_bkg_low > 0):
            bkg_mid_weight = n_bkg_low/n_bkg_mid
        else:
            bkg_mid_weight = 1.

        tot_bkg_weight = n_bkg_low + n_bkg_mid*bkg_mid_weight + n_bkg_high * bkg_high_weight

        sig_weight = tot_bkg_weight / n_sig

        print(n_bkg_low, n_bkg_mid, n_bkg_high, n_sig)
        print("TNT weights: bkg_high %.3f, bkg_mid %.3f, sig_weight %.3f"  %(bkg_high_weight, bkg_mid_weight, sig_weight))


        self.keys.add('weight' + extra_str)
        weights = np.ones_like(mjj, dtype=np.float32)
        weights[sig_cut] = sig_weight
        weights[bkg_cut & (mjj > mjj_high)] = bkg_high_weight

        self.f_storage.create_dataset('weight' + extra_str, data=weights)

        filter_frac = self.apply_mask(keep_mask)
        return filter_frac






    def make_ptrw(self, Y_key, use_weights = True, normalize = True, save_plots = False, plot_dir = "", extra_str = "", mjj_mid = -1., clip_perc = 1.):
       

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

        if(mjj_mid < 0):
            mjj_mid = (self.keep_mlow + self.keep_mhigh)/ 2.


        if(self.clip_pts): #restrict to bulk of distribution
            j1_pts = np.clip(j1_pts, np.percentile(j1_pts, clip_perc), np.percentile(j1_pts, 100. - clip_perc) )
            j2_pts = np.clip(j2_pts, np.percentile(j2_pts, clip_perc), np.percentile(j2_pts, 100. - clip_perc) )

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
        self.keys.add("j1_ptrw"+extra_str)

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
        self.keys.add("j2_ptrw" + extra_str)



    def __getitem__(self, key):
        if(not self.ready):
            print("Datareader has not loaded data yet! Must call read() first ")
            exit(1)
        if(key not in self.keys):
            print("Key %s not in list of preloaded keys!" % key, self.keys)
            exit(1)

        if(self.mask.shape[0] == 0):
            return self.f_storage[key][()]
        else:
            return self.f_storage[key][()][self.mask]

    def make_preprocessed_feats(self, key, qt):
        feats = self.f_storage[key]
        new_feats = qt.transform(feats)
        self.f_storage.create_dataset(key + "_normed", data = new_feats)
        self.keys.add(key + "_normed")

        


    def gen(self, key1, key2, key3 = None, batch_size = 256, key1_norm =None, key2_norm = None, key3_norm = None):
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


        h5_gen = MyGenerator(self.f_storage, n_objs, batch_size, key1, key2, key3, mask = mask_, 
                                                    key1_norm = key1_norm,key2_norm = key2_norm, key3_norm = key3_norm)
        return h5_gen

    def labeler_scores(self, model, key, chunk_size = 10000):
        #print(list(self.f_storage.keys()))
        #print("key", key)
        
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
            return filter_frac
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
def get_signal_mask(labels, mask, num_sig, rng):

    num_events = labels.shape[0]
    cur_sig =  np.sum(labels[mask] > 0)
    num_drop = int(cur_sig - num_sig)
    keep_idxs = np.array([True]*num_events)
    if(num_drop < 0): return keep_idxs
    all_idxs = np.arange(num_events, dtype=np.int32)
    sig_idxs = all_idxs[mask][labels[mask].reshape(-1) > 0]
    drop_sigs = rng.choice(sig_idxs, num_drop, replace = False)
    keep_idxs[drop_sigs] = False
    return keep_idxs


#create a mask that removes signal events to enforce a given fraction
#Random chance to keep each signal event

def get_signal_mask_rand(labels, mask, sig_frac, BB_seed=12345):
    print("Signal mass Rand DEPRECATED!?")

    np.random.seed(BB_seed)
    num_events = labels.shape[0]
    cur_frac =  np.mean(labels[mask] > 0)
    if(cur_frac <= sig_frac):
        return np.ones_like(labels)

    drop_frac = (1. - (sig_frac/cur_frac))
    rands = np.random.random(num_events)
    keep_idxs = (labels.reshape(-1) == 0) | (rands > drop_frac)
    return keep_idxs
