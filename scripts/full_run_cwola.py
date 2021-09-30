import sys
sys.path.append('..')
from utils.TrainingUtils import *
from cwola_ensemble import *
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
    options = parser.parse_args()

    if(options.output[-1] != '/'):
        options.output += '/'

    options.use_one = True

    subprocess.call("mkdir %s" % options.output, shell = True)

    if(options.numBatches % options.kfolds != 0):
        print("Number of batches (%i) must be multiple of number of kfolds (%i)" % (options.numBatches, options.kfolds))
        sys.exit(1)

    ksize = options.numBatches // options.kfolds

    batches_per_kfold = (options.kfolds -1 ) * ksize
    if(batches_per_kfold % options.lfolds !=0):
        print("Number of batches per kfold(%i) must be multiple of number of lfolds (%i)" % (batches_per_kfold, options.lfolds))
        sys.exit(1)

    options.num_val_batch = batches_per_kfold // options.lfolds

    total_batch_list = list(range(options.numBatches))

    fit_inputs = options.output + "fit_inputs.h5"
    base_path = os.path.abspath(".") + "/"
    print(base_path)
    merge_cmd = "python ../../CASEUtils/H5_maker/H5_merge.py %s "  % fit_inputs

    start_time = time.time()

    for k in range(2, options.kfolds):

        k_options = copy.deepcopy(options)
        k_list = copy.copy(total_batch_list)

        holdouts = list(range(k* ksize, (k+1) * ksize))

        for to_remove in holdouts:
            k_list.remove(to_remove)


        print(k_list) 
        k_options.data_batch_list = k_list

        #Do trainings
        k_options.output = options.output + "j1_kfold%i/" % k
        k_options.training_j = 1
        cwola_ensemble(k_options)
        k_options.output = options.output + "j2_kfold%i/" % k
        k_options.training_j = 2
        cwola_ensemble(k_options)


        #select events
        selection_options = copy.deepcopy(options)
        selection_options.data_batch_list = holdouts
        selection_options.val_batch_list = None
        selection_options.labeler_name = options.output + "{j_label}kfold%i/" % k
        selection_options.output = options.output + "fit_inputs_kfold%i.h5" % k
        merge_cmd += selection_options.output + " " 

        classifier_selection(selection_options)

    #merge different selections
    subprocess.call(merge_cmd ,shell = True)


    #Do fit
    fit_cmd = ("cd ../fitting; source deactivate;" 
              "eval `scramv1 runtime -sh`; python dijetfit.py -i %s -p %s; cd -; source deactivate; source activate mlenv0" % (base_path + fit_inputs, base_path + options.output))
    print(fit_cmd)
    subprocess.call(fit_cmd,  shell = True, executable = '/bin/bash')

    stop_time = time.time()
    print("Total time taken was %s min" % ( stop_time - start_time) / 60.)


