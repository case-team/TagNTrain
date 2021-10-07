import sys
import os
sys.path.append('..')
from utils.TrainingUtils import *
from create_model_ensemble import *
from classifier_selection import *
import subprocess
import h5py
import time


if(__name__ == "__main__"):
    parser = input_options()
    parser.add_argument("--effs", nargs="+", default = [], type = float)
    parser.add_argument("--kfolds", default = 5, type = int)
    parser.add_argument("--lfolds", default = 4, type = int)
    parser.add_argument("--numBatches", default = 40, type = int)
    parser.add_argument("--do_TNT",  default=False, action = 'store_true',  help="Use TNT (default cwola)")
    parser.add_argument("--condor", default = False, action = 'store_true')
    parser.add_argument("--step", default = "train",  help = 'Which step to perform (train, get, select, fit, roc, all)')
    parser.add_argument("--reload", default = False, action = 'store_true', help = "Reload based on previously saved options")
    options = parser.parse_args()

    if(options.output[-1] != '/'):
        options.output += '/'

    if(not os.path.exists(options.output)):
        subprocess.call("mkdir %s" % options.output, shell = True)

    if(options.reload):
        if(not os.path.exists(options.output + "run_opts.pkl")):
            print("Reload options specified but file %s doesn't exist. Exiting" % (options.output+"run_opts.pkl"))
            sys.exit(1)
        else:
            rel_opts = get_options_from_pkl(options.output + "run_opts.pkl")
            rel_opts.step = options.step
            if(len(options.effs) >0): rel_opts.effs = options.effs
            options = rel_opts
    else:
        #save options
        options_dict = options.__dict__
        write_options_to_pkl(options_dict, options.output + "run_opts.pkl" )


 



    if(options.numBatches % options.kfolds != 0):
        print("Number of batches (%i) must be multiple of number of kfolds (%i)" % (options.numBatches, options.kfolds))
        sys.exit(1)

    ksize = options.numBatches // options.kfolds

    batches_per_kfold = (options.kfolds -1 ) * ksize
    if(batches_per_kfold % options.lfolds !=0):
        print("Number of batches per kfold(%i) must be multiple of number of lfolds (%i)" % (batches_per_kfold, options.lfolds))
        sys.exit(1)

    if(options.step not in ["train", "get", "select", "fit", "roc", "all"]):
        print("Invalid option %s" % options.step)
        sys.exit(1)

    #parse what to do 
    get_condor = do_train = do_selection = do_fit = False
    do_train = options.step == "train"
    get_condor = options.step == "get"
    do_selection = options.step == "select"
    do_fit = options.step == "fit"
    do_roc = options.step == "roc"
    if(options.step == "all"):
        do_train = do_selection = do_fit = do_roc = True

    get_condor = get_condor | (do_selection and options.condor)


    options.num_val_batch = batches_per_kfold // options.lfolds

    total_batch_list = list(range(options.numBatches))

    fit_inputs = options.output + "fit_inputs.h5"
    base_path = os.path.abspath(".") + "/"

    start_time = time.time()
    kfold_options = []

    for k in range(options.kfolds):

        k_options = copy.deepcopy(options)
        k_list = copy.copy(total_batch_list)

        k_options.holdouts = list(range(k* ksize, (k+1) * ksize))

        for to_remove in k_options.holdouts:
            k_list.remove(to_remove)


        k_options.data_batch_list = k_list

        kfold_options += [k_options]

    #Do trainings
    if(do_train):
        for k,k_options in enumerate(kfold_options):
            k_options.label = options.label + "_j1_kfold%i" % k
            k_options.output = options.output + "j1_kfold%i/" % k
            k_options.training_j = 1
            create_model_ensemble(k_options)

            k_options.label = options.label + "_j2_kfold%i" % k
            k_options.output = options.output + "j2_kfold%i/" % k
            k_options.training_j = 2
            create_model_ensemble(k_options)



    #get all the condor models
    if(get_condor and options.condor):
        for k,k_options in enumerate(kfold_options):

                c_opts = condor_options().parse_args([])
                c_opts.getEOS = True

                #c_opts.name = "j1_kfold%i" % k
                c_opts.name = options.label + "_j1_kfold%i" % k
                c_opts.outdir = options.output + "j1_kfold%i/" % k
                doCondor(c_opts)
                #c_opts.name = "j2_kfold%i" % k
                c_opts.name = options.label + "_j2_kfold%i" % k
                c_opts.outdir = options.output + "j2_kfold%i/" % k
                doCondor(c_opts)

    #select events
    if(do_selection):
        merge_cmd = "python ../../CASEUtils/H5_maker/H5_merge.py %s "  % fit_inputs
        for k,k_options in enumerate(kfold_options):
            selection_options = copy.deepcopy(k_options)
            selection_options.data_batch_list = k_options.holdouts
            selection_options.val_batch_list = None
            selection_options.labeler_name = k_options.output + "{j_label}_kfold%i/" % k
            selection_options.output = k_options.output + "fit_inputs_kfold%i.h5" % k
            selection_options.do_roc = True
            selection_options.num_models = options.lfolds
            merge_cmd += selection_options.output + " " 

            classifier_selection(selection_options)

        #merge different selections
        subprocess.call(merge_cmd ,shell = True)

    if(do_roc):
        sig_effs = []
        bkg_effs = []
        labels = [ "kfold %i" % k for  k in range(options.kfolds)]
        colors = ["g", "b", "r", "gray", "purple", "pink", "orange", "m", "skyblue", "yellow"]
        for k,k_options in enumerate(kfold_options):
            np_fname = k_options.output + "fit_inputs_kfold%i_effs.npz" % k
            np_file = np.load(np_fname)
            sig_effs.append(np_file["sig_eff"])
            bkg_effs.append(np_file["bkg_eff"])

        sic_fname = options.output + options.label + "_sic.png"
        make_sic_plot(sig_effs, bkg_effs, colors = colors, labels = labels, fname = sic_fname)


    if(do_fit):
        #Do fit
        fit_cmd = ("cd ../fitting; source deactivate;" 
                  "eval `scramv1 runtime -sh`; python dijetfit.py -i %s -p %s; cd -; source deactivate; source activate mlenv0" % (base_path + fit_inputs, base_path + options.output))
        print(fit_cmd)
        subprocess.call(fit_cmd,  shell = True, executable = '/bin/bash')

    stop_time = time.time()
    print("Total time taken was %s" % ( stop_time - start_time))


