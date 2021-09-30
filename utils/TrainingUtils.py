from __future__ import print_function, division
from .model_defs import * 
from .losses import *
from .PlotUtils import *
from .DataReader import *
from .ModelEnsemble import *
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.stats import entropy
from sklearn.utils import shuffle as sk_shuffle
import argparse

def input_options():
#input options for all the of the different scripts. Not all options are used for everything

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--fin", default='../data/jet_images.h5', help="Input file with data for training.")
    parser.add_argument("--plot_dir", default='../plots/', help="Directory to output plots")
    parser.add_argument("--model_dir", default='../models/', help="Directory to read in and output models")
    parser.add_argument("-o", "--output", default='test.h5', help="Output location")
    parser.add_argument("-l", "--labeler_name", default='', help="Name of model used for labeling")
    parser.add_argument("--model_start", default="", help="Starting point for model (empty string for new model)")
    parser.add_argument("--model_type", default=2, type=int,  help="Type of model: 0 CNN (one jet), 1 auto encoder, 2 dense (one jet), 3 CNN (both jets), 4 dense (both jets), 5 VAE")

    parser.add_argument("--iter", dest = "tnt_iter", type = int, default=0, help="What iteration of  the tag & train algorithm this is (Start = 0).")
    parser.add_argument("--num_epoch", type = int, default=100, help="How many epochs to train for")
    parser.add_argument("--data_start", type = int, default=0, help="What event to start with")
    parser.add_argument("--data_sop", type = int, default=-1, help="What event to stop on")
    parser.add_argument("--num_data", type = int, default=-1, help="How many events to train on")
    parser.add_argument("--batch_size", type=int, default=256, help="Size of mini-batchs used for training")
    parser.add_argument("--batch_start", type=int, default=-1, help="Train over multiple batches of dataset. Starting batch")
    parser.add_argument("--batch_stop", type=int, default=-1, help="Train over multiple batches of dataset. Stopping batch (inclusive)")
    parser.add_argument("--val_batch_start", type=int, default=-1, help="Batches to use for validation start")
    parser.add_argument("--val_batch_stop", type=int, default=-1, help="Batches to use for validation stop ")
    parser.add_argument("--no_minor_bkgs", default = False, action = "store_true", help="Exclude minor backgrounds from sample")

    parser.add_argument("--use_one", default = False, action = "store_true", help="Make a classifier for one jet instead of both")
    parser.add_argument("-j", "--training_j", type =int, default = 1, help="Which jet to make a classifier for (1 or 2)")
    parser.add_argument("--use_images", default = False, action = "store_true", help="Make a classifier using jet images as inputs")
    parser.add_argument("--keep_mlow", type=int, default = -1,  help="Low mjj value to keep in dataset")
    parser.add_argument("--keep_mhigh", type=int, default = -1, help="High mjj value to keep in dataset")
    parser.add_argument("--mjj_low", type=int, default = 2250,  help="Low mjj cut value")
    parser.add_argument("--mjj_high", type=int, default = 2750, help="High mjj cut value")
    parser.add_argument("--mjj_sig", type=int, default = 2500, help="Signal mass (used for signal filtering)")
    parser.add_argument("--d_eta", type=float, default = -1, help="Delta eta cut")
    parser.add_argument("--no_ptrw", default = False, action="store_true",  help="Don't reweight events to have matching pt distributions in sig-rich and bkg-rich samples")
    parser.add_argument("--no_sample_weights", default = False, action="store_true", help="Don't do weighting of different signal / bkg regions")

    parser.add_argument("--large", default = False, action = "store_true", help="Use larger NN archetecture")
    parser.add_argument("--sig_idx", type = int, default = 1,  help="What index of signal to use")
    parser.add_argument("-s", "--sig_frac", type = float, default = -1.,  help="Reduce signal to S/B in signal region (< 0 to not use )")
    parser.add_argument("--sig_per_batch", type = float, default = -1.,  help="Reduce signal to this number of events in each batch (< 0 to not use )")
    parser.add_argument("--hadronic_only",  default=False, action='store_true',  help="Filter out leptonic decays of signal")
    parser.add_argument("--seeds", nargs="+", type = int, default = [123456],  help="RNG seeds for models")
    parser.add_argument("--BB_seed", type = int, default = 123456,  help="RNG seed for dataset")
    parser.add_argument("--num_models", type = int, default = 1,  help="How many networks to train (if >1 will save the one with best validation loss)")
    parser.add_argument("--no_mjj_cut", default = False, action = "store_true", help="Don't require a mass window")


    parser.add_argument("--local_storage", default =False, action="store_true",  help="Store temp files locally not on gpuscratch")
    parser.add_argument("--sig_cut", type=int, default = 80,  help="What classifier percentile to use to define sig-rich region in TNT")
    parser.add_argument("--bkg_cut", type=int, default = 40,  help="What classifier percentile to use to define bkg-rich region in TNT")


    parser.add_argument("--ptsort", default = False, action="store_true",  help="Sort j1 and j2 by pt rather than by jet mass")
    parser.add_argument("--randsort", default = False, action="store_true",  help="Sort j1 and j2 randomly rather than by jet mass")
    return parser

def load_dataset_from_options(options):
    if(not hasattr(options, 'keep_mlow') or not hasattr(options, 'keep_mhigh')):
        options.keep_mlow = options.keep_mhigh = -1

    if(not hasattr(options, 'data_batch_list')):
        data_batch_list = None
        val_batch_list = None

        if(options.batch_start >=0 and options.batch_stop >= 0): 
            data_batch_list = list(range(options.batch_start, options.batch_stop + 1))

        if(options.val_batch_start >=0 and options.val_batch_stop >= 0): 
            if(data_batch_list is None):
                print("Must give data batch range if using validation batches")
                exit(1)
            val_batch_list = list(range(options.val_batch_start, options.val_batch_stop+1))

            for i in val_batch_list: #validation batch range takes priority over regular batches
                if i in data_batch_list:
                    data_batch_list.remove(i)
    else:
        data_batch_list = options.data_batch_list
        if(hasattr(options, 'val_batch_list')): val_batch_list = options.val_batch_list
        else: val_batch_list = None

    if(val_batch_list != None):
        opts_dict = vars(copy.deepcopy(options))
        opts_dict['batch_list'] = val_batch_list
        val_data = DataReader(**opts_dict)

        #val_data = DataReader(fin = options.fin, keys = options.keys, sig_idx = options.sig_idx, sig_frac = options.sig_frac, 
        #        data_start = options.data_start, data_stop = options.data_start + options.num_data, 
        #    keep_mlow = options.keep_mlow, keep_mhigh = options.keep_mhigh, batch_list = val_batch_list, hadronic_only = options.hadronic_only, 
        #    mjj_sig = options.mjj_sig, BB_seed = options.BB_seed, d_eta = options.d_eta, local_storage = options.local_storage, randsort = options.randsort, sig_per_batch = options.sig_per_batch)
        #val_data.read()
    else:
        val_data = None

    opts_dict = vars(copy.deepcopy(options))
    opts_dict['batch_list'] = data_batch_list
    data = DataReader(**opts_dict)
        

    #data = DataReader(fin = options.fin, keys = options.keys, sig_idx = options.sig_idx, sig_frac = options.sig_frac, 
    #    data_start = options.data_start, data_stop = options.data_start + options.num_data, 
    #    keep_mlow = options.keep_mlow, keep_mhigh = options.keep_mhigh, batch_list = data_batch_list, hadronic_only = options.hadronic_only, 
    #    mjj_sig = options.mjj_sig, BB_seed = options.BB_seed, d_eta = options.d_eta, local_storage = options.local_storage, randsort = options.randsort, sig_per_batch = options.sig_per_batch)

    data.read()

    return data, val_data


def bce(yhat, y, weights = None):
    if(weights is not None):
        weight_mean = weights.mean()
        return (-( weights * (y * np.log(yhat) + (1 - y) * np.log(1 - yhat))).mean())/weight_mean
    else:
        return -((y * np.log(yhat) + (1 - y) * np.log(1 - yhat))).mean()


def compute_effcut_metric(sig_scores, bkg_scores, eff = 0.01):
    percentile = 100. - 100.*eff
    sb_cut = np.percentile(bkg_scores,percentile)
    return np.mean(sig_scores > sb_cut)




class RocCallback(tf.keras.callbacks.Callback):
    def __init__(self,training_data,validation_data, extra_label = "", freq = 5):
        self.extra_label = extra_label
        self.freq = freq
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
        if(epoch % self.freq != 0):
            return
        roc_train = roc_val = 0.
        msg = "\r%s" % self.extra_label
        if(not self.skip_train):
            y_pred_train = self.model.predict_proba(self.x)
            mask = ~np.isnan(y_pred_train)
            if(y_pred_train[mask].shape[0] > 1000):
                roc_train = roc_auc_score(self.y[mask], y_pred_train[mask])
                phrase = " roc-auc_train: %s" % str(round(roc_train,4))
                msg += phrase
        if(not self.skip_val):
            y_pred_val = self.model.predict_proba(self.x_val)
            mask = ~np.isnan(y_pred_val)
            if(y_pred_val[mask].shape[0] > 1000):
                roc_val = roc_auc_score(self.y_val[mask], y_pred_val[mask])
                phrase = " roc-auc_val: %s" % str(round(roc_val,4))
                msg += phrase
        print(msg, end =100*' ' + '\n')
        #print('\r%s roc-auc_train: %s - roc-auc_val: %s' % (self.extra_label, str(round(roc_train,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def make_selection(j1_scores, j2_scores, percentile):
# make a selection with a given efficiency using both scores (and)

    n_points = 1000
    j1_qs = quantile_transform(j1_scores.reshape(-1,1)).reshape(-1)
    j2_qs = quantile_transform(j2_scores.reshape(-1,1)).reshape(-1)

    threshs = [thresh for thresh in np.arange(0., 1., 1./n_points)]

    combined_effs = np.array([np.mean((j1_qs > thresh & j2_qs > thresh)) for thresh in threshs])
    cut_idx = np.argwhere(combined_effs < (100. - percentile)/100.)[0][0]
    mask = (j1_qs > threshs[cut_idx]) & (j2_qs > threshs[cut_idx])
    print("Cut idx %i, eff %.3e, j1_cut %.3e, j2_cut %.3e " %(cut_idx, combined_effs[cut_idx]))
    print(np.mean(mask))
    return mask

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




def get_single_jet_scores(model_name, model_type, j_images=None, j_dense_inputs=None,  num_models = 1, batch_size = 512):

    if(model_type <= 2):
        j_model = ModelEnsemble(location = model_name, num_models = num_models)

        if(model_type == 0):  #cnn
            j_score = j_model.predict(j_images, batch_size = batch_size)
        elif(model_type == 1): #autoencoder
            j_reco_images = j_model.predict(j_images, batch_size=batch_size)
            j_score =  np.mean(np.square(j_reco_images - j_images), axis=(1,2))
        elif(model_type == 2): #dense
            j_score = j_model.predict(j_dense_inputs, batch_size = batch_size)
    elif(model_type == 5): #vae
        j_model = vae(0, model_dir = model_dir + "j_" +  model_name)
        j_model.load()
        j_reco_images, j_z_mean, j_z_log_var = j_model.predict_with_latent(j_images)
        j_score = compute_loss_of_prediction_mse_kl(j_images, j_reco_images, j_z_mean, j_z_log_var)[0]
    else:
        print("Wrong model type for jet_scores")
        return None

    return j_score.reshape(-1)






def get_jet_scores(model_dir, model_name, model_type, j1_images=None, j2_images=None, j1_dense_inputs=None, j2_dense_inputs=None, 
        num_models = 1, batch_size = 512):

    if(model_type <= 2):
        if(len(model_name) != 2):
            if('j_label' in model_name):
                j1_fname = model_dir + model_name.format(j_label = "j1")
                j2_fname = model_dir + model_name.format(j_label = "j2")
            else:
                j1_fname = j2_fname = model_dir + model_name
            #if('/' not in model_name):
            #    j1_fname = model_dir + "j1_" + model_name
            #    j2_fname = model_dir + "j2_" + model_name
            #else:
            #    ins_idx = model_name.rfind('/')+1
            #    j1_fname = model_dir + model_name[:ins_idx] + "j1_" + model_name[ins_idx:]
            #    j2_fname = model_dir + model_name[:ins_idx] + "j2_" + model_name[ins_idx:]
            print(j1_fname, j2_fname)
            j1_model = ModelEnsemble(location = j1_fname, num_models = num_models)
            j2_model = ModelEnsemble(location = j2_fname, num_models = num_models)
        else:
            j1_model = ModelEnsemble(location = model_dir + model_name[0], num_models = num_models)
            j2_model = ModelEnsemble(location =  model_dir + model_name[1], num_models = num_models)

        if(model_type == 0):  #cnn
            j1_score = j1_model.predict(j1_images, batch_size = batch_size)
            j2_score = j2_model.predict(j2_images, batch_size = batch_size)
        elif(model_type == 1): #autoencoder
            j1_reco_images = j1_model.predict(j1_images, batch_size=batch_size)
            j1_score =  np.mean(np.square(j1_reco_images - j1_images), axis=(1,2))
            j2_reco_images = j2_model.predict(j2_images, batch_size=batch_size)
            j2_score =  np.mean(np.square(j2_reco_images -  j2_images), axis=(1,2))
        elif(model_type == 2): #dense
            j1_score = j1_model.predict(j1_dense_inputs, batch_size = batch_size)
            j2_score = j2_model.predict(j2_dense_inputs, batch_size = batch_size)
    elif(model_type == 5): #vae
        j1_model = vae(0, model_dir = model_dir + "j1_" +  model_name)
        j1_model.load()
        j1_reco_images, j1_z_mean, j1_z_log_var = j1_model.predict_with_latent(j1_images)
        j1_score = compute_loss_of_prediction_mse_kl(j1_images, j1_reco_images, j1_z_mean, j1_z_log_var)[0]
        j2_model = vae(0, model_dir = model_dir + "j2_" +  model_name)
        j2_model.load()
        j2_reco_images, j2_z_mean, j2_z_log_var = j2_model.predict_with_latent(j2_images)
        j2_score = compute_loss_of_prediction_mse_kl(j2_images, j2_reco_images, j2_z_mean, j2_z_log_var)[0]
    else:
        print("Wrong model type for jet_scores")
        return None

    return j1_score.reshape(-1), j2_score.reshape(-1)

def get_jj_scores(model_dir, model_name, model_type, jj_images = None, jj_dense_inputs = None, batch_size = 512, num_models = 1):
    if(model_type == 3): #CNN both jets
        jj_model = tf.keras.models.load_model(model_dir + model_name)
        scores = jj_model.predict(jj_images, batch_size = batch_size).reshape(-1)

    elif(model_type == 4): #Dense both jets
        jj_model = tf.keras.models.load_model(model_dir + model_name)
        scores = jj_model.predict(jj_dense_inputs, batch_size = batch_size).reshape(-1)
    else:
        print("Wrong model type for jj scores!")
        return None

    return scores

