import sys
sys.path.append('..')
from training.train_cwola_hunting_network import *
from training.tag_and_train import *



def create_model_ensemble(options):
    if(hasattr(options, "data_batch_list")):
        num_total_batches = len(options.data_batch_list)

    else:
        num_total_batches = (options.batch_stop - options.batch_start)+1
    if(num_total_batches % options.num_val_batch != 0):
        print("Total number of batches (%i)  not a multiple of validation batches (%i). Splitting unclear" % (num_total_batches, options.num_val_batch))
        exit(1)

    if(options.output[-1] != '/'):
        options.output += '/'

    num_ensemble_models = num_total_batches // options.num_val_batch
    print("Will train %i models for the ensemble. %i total batches, %i validation batchsize" % (num_ensemble_models, num_total_batches, options.num_val_batch))
    print("Will save to %s" % options.output)
    if('{seed}' in options.output):
        for seed in options.seeds:
            dirname = options.output.format(seed = seed)
            os.system("mkdir %s" % dirname)
    else:
        os.system("mkdir %s" % options.output)

    if(options.condor):
        condor_dir = options.output + "condor/"
        os.system("mkdir " +  condor_dir)

    for i in range(num_ensemble_models):
        options_copy = copy.deepcopy(options)
        options_copy.output += "model%i.h5" % i

        options_copy.val_batch_start = i*options.num_val_batch
        options_copy.val_batch_stop = (i+1)*options.num_val_batch - 1 

        if(hasattr(options_copy, "data_batch_list")):
            options_copy.val_batch_list = options_copy.data_batch_list[options_copy.val_batch_start : options_copy.val_batch_stop+1]

            for i in options_copy.val_batch_list: #validation batch range takes priority over regular batches
                if i in options_copy.data_batch_list:
                    options_copy.data_batch_list.remove(i)
            print('data', options_copy.data_batch_list)
            print('val', options_copy.val_batch_list)


        if(not options.condor):
            if(options.do_TNT): tag_and_train(options_copy)
            else: train_cwola_hunting_network(options_copy)
        else:

            options_dict = options_copy.__dict__

            with open(condor_dir + "train_params%i.pkl" % i , "w") as f:
                pickle.dump(options_dict, f)

            condor_cmd = "python ../scripts/train_from_param_dict.py train_params%i.pkl" % i


if(__name__ == "__main__"):
    parser = input_options()
    parser.add_argument("--num_val_batch", type=int, default=5, help="How many batches to use for validation")
    parser.add_argument("--do_TNT",  default=False, action = 'store_true',  help="Use TNT (default cwola)")
    parser.add_argument("--condor",  default=False, action = 'store_true',  help="Submit all NN trainings to condor")
    options = parser.parse_args()
    create_model_ensemble(options)
