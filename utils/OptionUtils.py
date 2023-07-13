from .DataReader import *
import pickle
import sys
import argparse
import json


def input_options():
#input options for all the of the different scripts. Not all options are used for everything

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--fin", default='../data/jet_images.h5', help="Input file with data for training.")
    parser.add_argument("--plot_dir", default='../plots/', help="Directory to output plots")
    parser.add_argument("--label", default='', help="extra str for labeling output")
    parser.add_argument("--model_dir", default='../models/', help="Directory to read in and output models")
    parser.add_argument("-o", "--output", default='test.h5', help="Output location")
    parser.add_argument("-l", "--labeler_name", default='', help="Name of model used for labeling")
    parser.add_argument("--model_start", default="", help="Starting point for model (empty string for new model)")
    parser.add_argument("--model_type", default=2, type=int,  help="Type of model: 0 CNN (one jet), 1 auto encoder, 2 dense (one jet), 3 CNN (both jets), 4 dense (both jets), 5 VAE")

    parser.add_argument("--iter", dest = "tnt_iter", type = int, default=0, help="What iteration of  the tag & train algorithm this is (Start = 0).")
    parser.add_argument("--num_epoch", type = int, default=100, help="How many epochs to train for")
    parser.add_argument("--data_start", type = int, default=0, help="What event to start with")
    parser.add_argument("--data_stop", type = int, default=-1, help="What event to stop on")
    parser.add_argument("--num_data", type = int, default=-1, help="How many events to train on")
    parser.add_argument("--max_events", type = int, default=-1, help="Max number of events ")
    parser.add_argument("--val_max_events", type = int, default=-1, help="Max number of events for val sample")

    parser.add_argument("--batch_size", type=int, default=256, help="Size of mini-batchs used for training")
    parser.add_argument("--batch_start", type=int, default=-1, help="Train over multiple batches of dataset. Starting batch")
    parser.add_argument("--batch_stop", type=int, default=-1, help="Train over multiple batches of dataset. Stopping batch (inclusive)")
    parser.add_argument("--val_batch_start", type=int, default=-1, help="Batches to use for validation start")
    parser.add_argument("--val_batch_stop", type=int, default=-1, help="Batches to use for validation stop ")
    parser.add_argument("--no_minor_bkgs", default = False, action = "store_true", help="Exclude minor backgrounds from sample")
    parser.add_argument("--preprocess", default = "", help="Apply a transformation for preprocessing inputs (normal or uniform)")
    parser.add_argument("--keep_tau1", default = False, action = 'store_true',  help="Use tau1 feature")

    parser.add_argument("--use_one", type = bool, default = True, help="Make a classifier for one jet instead of both")
    parser.add_argument("-j", "--training_j", type =int, default = 1, help="Which jet to make a classifier for (1 or 2)")
    parser.add_argument("--use_images", default = False, action = "store_true", help="Make a classifier using jet images as inputs")

    parser.add_argument("--keep_mlow", type=int, default = -1,  help="Low mjj value to keep in dataset")
    parser.add_argument("--keep_mhigh", type=int, default = -1, help="High mjj value to keep in dataset")
    parser.add_argument("--mjj_low", type=int, default = -1,  help="Low mjj cut value")
    parser.add_argument("--mjj_high", type=int, default = -1, help="High mjj cut value")
    parser.add_argument("--mjj_sig", type=int, default = 2500, help="Signal mass (used for signal filtering only)")
    parser.add_argument("--mbin", type=int, default = -1, help="Mass bin based on mjj binning scheme")

    parser.add_argument("--deta", type=float, default = -1, help="Delta eta cut")
    parser.add_argument("--deta_min", default = -1, type = float, help = "Minimum dEta")
    parser.add_argument("--sideband", default = False, action = 'store_true', help = "Add cuts for sideband")
    parser.add_argument("--data", default = False, action = 'store_true', help = "Is data")

    parser.add_argument("--no_ptrw", default = False, action="store_true",  help="Don't reweight events to have matching pt distributions in sig-rich and bkg-rich samples")
    parser.add_argument("--no_sample_weights", default = False, action="store_true", help="Don't do weighting of different signal / bkg regions")

    parser.add_argument("--large_net", default = False, action = "store_true", help="Use larger NN archetecture")
    parser.add_argument("--small_net", default = False, action = "store_true", help="Use smaller NN archetecture")
    parser.add_argument("--sig_idx", type = int, default = 1,  help="What index of signal to use")
    parser.add_argument("--sig_file", type = str, default = "",  help="Load signal from separate file")
    parser.add_argument("--replace_ttbar", default = False, action = "store_true", help = "Filter out ttbar events from the dataset (to replace with separate sample")
    parser.add_argument("--sig_weights", default = True, action = "store_true",  help="Use weighted random sampling ( based on SF's) for signal file (only matters for separate sig_file used)")
    parser.add_argument("--lund_weights", default = False, action = "store_true",  help="Use Lund Weights for signal file")
    parser.add_argument("--no_lund_weights", dest='lund_weights', action = "store_false",  help="Don't use Lund Weights for signal file")
    parser.add_argument("--sig_sys", type= str, default = "",  help="Use weights (SF's) for signal file")
    parser.add_argument("-s", "--sig_frac", type = float, default = -1.,  help="Reduce signal to S/B in signal region (< 0 to not use )")
    parser.add_argument("--sig_per_batch", type = float, default = -1.,  help="Reduce signal to this number of events in each batch (< 0 to not use )")
    parser.add_argument("--spb_before_selection",  default = False, action = 'store_true',  help="Whether the of sig events per batch refers to number before mass and delta eta cuts (or not)")
    parser.add_argument("--hadronic_only",  default=False, action='store_true',  help="Filter out leptonic decays of signal")
    parser.add_argument("--seed", type = int, default = 123456,  help="RNG seeds for models")
    parser.add_argument("--BB_seed", type = int, default = 123456,  help="RNG seed for dataset")
    parser.add_argument("--num_models", type = int, default = 1,  help="How many networks to train (if >1 will save the one with best validation loss)")
    parser.add_argument("--no_mjj_cut", default = False, action = "store_true", help="Don't require a mass window")
    parser.add_argument("--save_mem", default = False, action = "store_true", help="Delete BB files in condor jobs after reading them")

    parser.add_argument("--keep_LSF",  action = "store_true", help="Keep LSF for dense inputs")
    parser.add_argument("--no_LSF", dest = 'keep_LSF', action = "store_false", help="Dont Keep LSF for dense inputs")
    parser.set_defaults(keep_LSF=True)
    parser.add_argument("--clip_feats", action = "store_true", help="Clip input feature values to avoid negatives")
    parser.add_argument("--no_clip_feats", dest = 'clip_feats', action = "store_false", help="Clip input feature values to avoid negatives")
    parser.set_defaults(clip_feats=True)
    parser.add_argument("--clip_pts", action = "store_true", help="Clip pts when doing reweighting to avoid outliers")
    parser.add_argument("--no_clip_pts", dest = 'clip_pts', action = "store_false", help="Clip pts when doing reweighting to avoid outliers")
    parser.set_defaults(clip_pts=True)

    parser.add_argument("--nsubj_ratios", dest = 'nsubj_ratios', action = "store_true", help="Nsubj ratios as input features")
    parser.add_argument("--no_nsubj_ratios", dest = 'nsubj_ratios', action = "store_false", help="Clip input feature values to avoid negatives")
    parser.set_defaults(nsubj_ratios=True)

    parser.add_argument("--eff_only", default = False,  action = 'store_true', help = 'When merging/selecting only saving efficiencies, not fit inputs')

    parser.add_argument("--local_storage", default =False, action="store_true",  help="Store temp files locally not on gpuscratch")
    parser.add_argument("--sig_cut", type=int, default = 80,  help="What classifier percentile to use to define sig-rich region in TNT")
    parser.add_argument("--bkg_cut", type=int, default = 50,  help="What classifier percentile to use to define bkg-rich region in TNT")

    parser.add_argument("--eff_cut", type=float, default = 0.01,  help="What classifier percentile to use for eff_cut computation")

    parser.add_argument("--ptsort", default = False, action="store_true",  help="Sort j1 and j2 by pt rather than by jet mass")
    parser.add_argument("--randsort", default = False, action="store_true",  help="Sort j1 and j2 randomly rather than by jet mass")

    parser.add_argument("--score_comb", default = "mult", help = 'How to combine the anomaly scores from two jets into one')

    parser.add_argument("--TNT_bkg_cut", default = 3, type = int,  
            help="What type of mass window for bkg-like sample (0: AE cut and SB, 1:AE cut only, SB and SR, 2: AE cut and SR, 3: AE cut or SR)")
    parser.add_argument("--AE_size", default = 6, type = int,  help="Size of AE latent space")
    parser.add_argument("--dense_AE", default =False, action = 'store_true')
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
        opts_dict['max_events'] = options.val_max_events
        val_data = DataReader(**opts_dict)

        #val_data = DataReader(fin = options.fin, keys = options.keys, sig_idx = options.sig_idx, sig_frac = options.sig_frac, 
        #        data_start = options.data_start, data_stop = options.data_start + options.num_data, 
        #    keep_mlow = options.keep_mlow, keep_mhigh = options.keep_mhigh, batch_list = val_batch_list, hadronic_only = options.hadronic_only, 
        #    mjj_sig = options.mjj_sig, BB_seed = options.BB_seed, deta = options.deta, local_storage = options.local_storage, randsort = options.randsort, sig_per_batch = options.sig_per_batch)
        val_data.read()
    else:
        val_data = None

    opts_dict = vars(copy.deepcopy(options))
    opts_dict['batch_list'] = data_batch_list
    data = DataReader(**opts_dict)
        

    #data = DataReader(fin = options.fin, keys = options.keys, sig_idx = options.sig_idx, sig_frac = options.sig_frac, 
    #    data_start = options.data_start, data_stop = options.data_start + options.num_data, 
    #    keep_mlow = options.keep_mlow, keep_mhigh = options.keep_mhigh, batch_list = data_batch_list, hadronic_only = options.hadronic_only, 
    #    mjj_sig = options.mjj_sig, BB_seed = options.BB_seed, deta = options.deta, local_storage = options.local_storage, randsort = options.randsort, sig_per_batch = options.sig_per_batch)

    data.read()

    return data, val_data

def load_signal_file(options):
    if(not hasattr(options, 'data_batch_list')):
        data_batch_list = None
        if(options.batch_start >=0 and options.batch_stop >= 0): 
            data_batch_list = list(range(options.batch_start, options.batch_stop + 1))

    else:
        data_batch_list = options.data_batch_list
    s_opts = copy.deepcopy(options)
    s_opts.deta = s_opts.deta_min = s_opts.keep_mlow = s_opts.keep_mhigh = -1.
    s_opts.sig_per_batch = s_opts.sig_frac = -1
    s_opts.sig_only = True
    s_opts.keys += ["jet_kinematics", "sys_weights", "j1_JME_vars", "j2_JME_vars" ]
    if(s_opts.lund_weights):
        s_opts.keys += ['lund_weights', 'lund_weights_stat_var', 'lund_weights_pt_var', 'lund_weights_sys_var']
    #s_opts.keys += ["jet_kinematics" ]

    s_opts_dict = vars(s_opts)
    s_opts_dict['batch_list'] = data_batch_list
    s_data = DataReader(**s_opts_dict)
    s_data.read()
    return s_data
    
def lookup_mjj_bins(mbin):
        if(mbin > 10):
            mbins = mass_bins2
            mbin_idx = mbin - 10
        else:
            mbins = mass_bins1
            mbin_idx = mbin
        mjj_low = mbins[mbin_idx]
        mjj_high = mbins[mbin_idx+1]
        sb_low = mbins[mbin_idx-1]
        sb_high = mbins[mbin_idx+2]
        return sb_low, mjj_low, mjj_high, sb_high


def compute_mjj_window(options):
    if(options.mbin >= 0): # use predefined binning
        options.keep_mlow, options.mjj_low, options.mjj_high, options.keep_mhigh = lookup_mjj_bins(options.mbin)
        options.eff_cut = mass_bin_select_effs[options.mbin] / 100.
        options.sig_cut = mass_bin_TNT_cuts[options.mbin]



    else: #compute 

        window_size = (options.mjj_high - options.mjj_low)/2.
        window_frac = window_size / ((options.mjj_high + options.mjj_low)/ 2.)

        window_low_size = window_frac*options.mjj_low / (1 + window_frac)
        window_high_size = window_frac*options.mjj_high / (1 - window_frac)
        options.keep_mlow = options.mjj_low - window_low_size
        options.keep_mhigh = options.mjj_high + window_high_size

class OptStruct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def get_options_from_pkl(fname):

    with open(fname, "rb") as f:
        options_dict = pickle.load(f)
    options = OptStruct(**options_dict)

    return options


def get_options_from_json(fname):

    with open(fname, "rb") as f:
        options_dict = json.load(f, encoding = 'latin-1')
    options = OptStruct(**options_dict)

    return options

def write_options_to_pkl(options, fname, write_mode = "wb"):
    with open(fname, write_mode) as f:
        pickle.dump(options, f)
    return

def write_options_to_json(options, fname, write_mode = "w"):
    with open(fname, write_mode) as f:
        json.dump(options, f)
    return



def get_params(fname):
    with open(fname, "r") as f:
        pickle.dump(results, f)


def write_params(fname, params):
    with open(fname, "w") as f:
        json.dump(params, f )
