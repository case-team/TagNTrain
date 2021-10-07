from .DataReader import *
import pickle
import sys
import argparse


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
    parser.add_argument("--data_sop", type = int, default=-1, help="What event to stop on")
    parser.add_argument("--num_data", type = int, default=-1, help="How many events to train on")
    parser.add_argument("--batch_size", type=int, default=256, help="Size of mini-batchs used for training")
    parser.add_argument("--batch_start", type=int, default=-1, help="Train over multiple batches of dataset. Starting batch")
    parser.add_argument("--batch_stop", type=int, default=-1, help="Train over multiple batches of dataset. Stopping batch (inclusive)")
    parser.add_argument("--val_batch_start", type=int, default=-1, help="Batches to use for validation start")
    parser.add_argument("--val_batch_stop", type=int, default=-1, help="Batches to use for validation stop ")
    parser.add_argument("--no_minor_bkgs", default = False, action = "store_true", help="Exclude minor backgrounds from sample")

    parser.add_argument("--use_one", type = bool, default = True, help="Make a classifier for one jet instead of both")
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
        val_data.read()
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

class OptStruct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def get_options_from_pkl(fname):

    with open(fname, "rb") as f:
        options_dict = pickle.load(f)
    options = OptStruct(**options_dict)

    return options

def write_options_to_pkl(options, fname):
    with open(fname, "wb") as f:
        pickle.dump(options, f)
    return




