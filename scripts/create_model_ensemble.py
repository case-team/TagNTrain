import sys
sys.path.append('..')
from training.train_cwola_hunting_network import *
from training.tag_and_train import *
from condor.doCondor import *



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
        dirname = options.output.format(seed = options.seed)
        os.system("mkdir %s" % dirname)
    else:
        if(not os.path.exists(options.output)): os.system("mkdir %s" % options.output)

    opts_list = []
    for i in range(num_ensemble_models):
        options_copy = copy.deepcopy(options)
        options_copy.output += "model%i.h5" % i

        options_copy.val_batch_start = i*options.num_val_batch
        options_copy.val_batch_stop = (i+1)*options.num_val_batch - 1 

        if(hasattr(options_copy, "data_batch_list")):
            options_copy.val_batch_list = options_copy.data_batch_list[options_copy.val_batch_start : options_copy.val_batch_stop+1]

            for j in options_copy.val_batch_list: #validation batch range takes priority over regular batches
                if j in options_copy.data_batch_list:
                    options_copy.data_batch_list.remove(j)
            #print(i, 'data', options_copy.data_batch_list)
            #print(i, 'val', options_copy.val_batch_list)


        if(not options.condor):
            if(options.do_TNT): tag_and_train(options_copy)
            else: train_cwola_hunting_network(options_copy)

        else:

            options_copy.output = "model%i.h5" % i
            options_copy.local_storage = True
            options_copy.fin = "../data/BB/"
            options_dict = options_copy.__dict__

            write_options_to_json(options_dict, "train_opts_%i.json" % i )
            opts_list.append("train_opts_%i.json" % i)

    if(options.condor):
        condor_dir = options.output + "condor_jobs/"
        if( not os.path.exists(condor_dir)): os.system("mkdir %s" % condor_dir)
        c_opts = condor_options().parse_args([])
        c_opts.nJobs = num_ensemble_models
        c_opts.outdir = condor_dir
        base_script = "../condor/scripts/train_from_json.sh"
        train_script = condor_dir + "train_script.sh"
        if(options.fin[-1] == "/"):
            bb_name = options.fin.split("/")[-2]
        else:
            bb_name = options.fin.split("/")[-1]
        
        os.system("cp %s %s" % (base_script, train_script))
        os.system("sed -i 's/BB_NAME/%s/g' %s" % (bb_name, train_script))
        if(len(options.sig_file) > 0):
            sig_fn = options.sig_file.split("/")[-1]
            os.system("sed -i 's/SIGFILE/%s/g' %s" % ("--sig_file " + sig_fn, train_script))
        else:
            os.system("sed -i 's/SIGFILE//g' %s" % (train_script))
        c_opts.script = train_script
        c_opts.name = options.label
        c_opts.sub = True
        c_opts.overwrite = True
        #c_opts.sub = False
        c_opts.input = opts_list
        doCondor(c_opts)
        for fname in opts_list:
            os.system("rm %s" % fname)



if(__name__ == "__main__"):
    parser = input_options()
    parser.add_argument("--num_val_batch", type=int, default=5, help="How many batches to use for validation")
    parser.add_argument("--do_TNT",  default=False, action = 'store_true',  help="Use TNT (default cwola)")
    parser.add_argument("--do_ae",  default=False, action = 'store_true',  help="Train autoencoder (default cwola)")
    parser.add_argument("--do_ttbar",  default=False, action = 'store_true',  help="Do ttbar CR training")
    parser.add_argument("--condor",  default=False, action = 'store_true',  help="Submit all NN trainings to condor")
    options = parser.parse_args()
    create_model_ensemble(options)
