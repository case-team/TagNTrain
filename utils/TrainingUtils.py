from __future__ import print_function, division
from .model_defs import * 
from .PlotUtils import *
import tensorflow as tf
from sklearn.model_selection import train_test_split
import h5py
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.stats import entropy
from sklearn.utils import shuffle as sk_shuffle
from keras import optimizers, callbacks, losses
from keras.callbacks import Callback

class RocCallback(Callback):
    def __init__(self,training_data,validation_data, extra_label = ""):
        self.extra_label = extra_label
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.skip_val = self.skip_train = False
        if(np.mean(self.y_val) < 1e-5):
            print("Not enough signal in validation set, will skip auc")
            self.skip_val = True
        if(np.mean(self.y) < 1e-5):
            print("Not enough signal in train set, will skip auc")
            self.skip_train = True


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        roc_train = roc_val = 0.
        if(not self.skip_train):
            y_pred_train = self.model.predict_proba(self.x)
            roc_train = roc_auc_score(self.y, y_pred_train)
        if(not self.skip_val):
            y_pred_val = self.model.predict_proba(self.x_val)
            roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\r%s roc-auc_train: %s - roc-auc_val: %s' % (self.extra_label, str(round(roc_train,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def print_signal_fractions(y_true, y):
    #compute true signal fraction in signal-rich region
    y_true = y_true.reshape(-1)
    y = y.reshape(-1)
    true_sigs = (y_true > 0.9 ) & (y > 0.9)
    lost_sigs = (y_true > 0.9) & (y < 0.1 )
    #print(true_sigs.shape, lost_sigs.shape, y.shape)
    sig_frac = np.mean(true_sigs) / np.mean(y)
    outside_frac = np.mean(lost_sigs)/np.mean(1-np.mean(y))
    SR_frac = np.mean(y)
    print("Signal-rich region as a fraction of total labeled events is %.4f. Sig frac in SR is %.4f \n" % (SR_frac, sig_frac))
    print("Sig frac in bkg_region is %.4f \n" %outside_frac)
    #print("Overall signal fraction is %.4f \n" %(mass_frac * frac + (1-mass_frac)*outside_frac))



def sample_split(*args, **kwargs):
    sig_region_cut = kwargs.pop('sig_cut', 0.9)
    bkg_region_cut = kwargs.pop('bkg_cut', 0.2)
    cut_var = kwargs.pop('cut_var', np.array([]))
    sig_high = kwargs.pop('sig_high', True)

    if len(kwargs):
        raise TypeError('following kwargs are invalid: {}'.format(kwargs))

    assert len(args) > 0

    if(cut_var.size == 0):
        raise TypeError('Must supply cut_var argument!')

    #sig_high is whether signal lives at > cut value or < cut value
    if(sig_high):
        sig_cut = cut_var > sig_region_cut
        bkg_cut = cut_var < bkg_region_cut
    else:
        sig_cut = cut_var < sig_region_cut
        bkg_cut = cut_var > bkg_region_cut



    args_sig = [x[sig_cut] for x in args]
    args_bkg = [x[bkg_cut] for x in args]



    args_zipped = [np.concatenate((args_sig[i], args_bkg[i])) for i in range(len(args))]
    labels = np.concatenate((np.ones((args_sig[0].shape[0]), dtype=np.float32), np.zeros((args_bkg[0].shape[0]), dtype=np.float32)))
    
    do_shuffle = True

    if(do_shuffle):
        shuffled = sk_shuffle(*args_zipped, labels, random_state = 123)
        args_shuffled = shuffled[:-1]
        labels = shuffled[-1]
        return args_shuffled, labels

    else:
        return args_zipped, labels




#helper to read h5py mock-datasets
def prepare_dataset(fin, signal_idx =1, keys = ['j1_images', 'j2_images', 'mjj'], sig_frac = -1., start = 0, stop = -1):
    print("Loading file %s \n" % fin)
    if(stop > 0): print("Selecting events %i to %i \n" % (start, stop))
    data = dict()
    f = h5py.File(fin, "r")
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
            data['mjj'] = f["event_info"][start:stop][mask,0]
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



#taken from https://stackoverflow.com/questions/47731935/using-multiple-validation-sets-with-keras
class AdditionalValidationSets(tf.keras.callbacks.Callback):
    def __init__(self, validation_sets, verbose=0, batch_size=None):
        """
        :param validation_sets:
        a list of 3-tuples (validation_data, validation_targets, validation_set_name)
        or 4-tuples (validation_data, validation_targets, sample_weights, validation_set_name)
        :param verbose:
        verbosity mode, 1 or 0
        :param batch_size:
        batch size to be used when evaluating on the additional datasets
        """
        super(AdditionalValidationSets, self).__init__()
        self.validation_sets = validation_sets
        for validation_set in self.validation_sets:
            if len(validation_set) not in [2, 3]:
                raise ValueError()
        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.batch_size = batch_size

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        for validation_set in self.validation_sets:
            if len(validation_set) == 3:
                validation_data, validation_targets, validation_set_name = validation_set
                sample_weights = None
            elif len(validation_set) == 4:
                validation_data, validation_targets, sample_weights, validation_set_name = validation_set
            else:
                raise ValueError()

            results = self.model.evaluate(x=validation_data,
                    y=validation_targets,
                    verbose=self.verbose,
                    sample_weight=sample_weights,
                    batch_size=self.batch_size)

            print("\n")
            for i, result in enumerate(results):
                if i == 0:
                    valuename = validation_set_name + '_loss'
                else:
                    valuename = validation_set_name + '_' + self.model.metrics[i-1]
                print("%s   %.4f " % (valuename, result))
                self.history.setdefault(valuename, []).append(result)
            print("\n")
