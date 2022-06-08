from full_run import *
from train_from_param_dict import *

def train_ae_mbin(options, idx, k = 0, output = ""):
    print("mbin %i" % options.mbin)

    if(options.output[-1] != '/'):
        options.output += '/'

    if(not os.path.exists(options.output)): os.system("mkdir %s" % options.output)


    if(not options.condor):
        print("training locally")
        options.output = "model%i.h5" % idx
        train_autoencoder(options)
        cmd = "mv model%i.h5 %s/models/jrand_AE_kfold%i_mbin%i.h5" % (idx, output, k, options.mbin)
        print(cmd)
        os.system(cmd)

    else:

        outdir = options.output
        options.output = "model%i.h5" % idx
        options.local_storage = True
        options.fin = "../data/BB/"
        options_dict = options.__dict__

        write_options_to_json(options_dict, outdir + "train_opts_%i.json" % idx )




def train_all_aes(options):
    if(len(options.label) == 0):
        options.label = options.output.split("/")[-1]


    if(options.output[-1] != '/'):
        options.output += '/'

    if(not os.path.exists(options.output)):
        subprocess.call("mkdir %s" % options.output, shell = True)

    if(options.reload):

        if(not os.path.exists(options.output + "run_opts.json")):
            print("Reload options specified but file %s doesn't exist (add --new to create new directory). Exiting" % (options.output+"run_opts.json"))
            sys.exit(1)
        else:
            rel_opts = get_options_from_json(options.output + "run_opts.json")
            print(rel_opts.__dict__)
            rel_opts.keep_LSF = options.keep_LSF #quick fix
            rel_opts.num_events = options.num_events #quick fix
            rel_opts.recover = options.recover
            rel_opts.condor = options.condor
            if(abs(options.mjj_sig - 2500.) > 1.):  rel_opts.mjj_sig  = options.mjj_sig #quick fix
            rel_opts.step = options.step
            if(len(options.effs) >0): rel_opts.effs = options.effs
            if(options.sig_per_batch >= 0): rel_opts.sig_per_batch = options.sig_per_batch

            options = rel_opts

    options.randsort = True
    options.training_j = 1
    options.do_ae = True
    options.do_TNT = False

    options.num_models = 1

    if(options.numBatches % options.kfolds != 0):
        print("Number of batches (%i) must be multiple of number of kfolds (%i)" % (options.numBatches, options.kfolds))
        sys.exit(1)

    ksize = options.numBatches // options.kfolds

    batches_per_kfold = (options.kfolds -1 ) * ksize

    #save options
    options_dict = options.__dict__
    write_options_to_json(options_dict, options.output + "run_opts.json" )


    #read saved parameters
    if(os.path.exists(options.output + "saved_params.json")):
        with open(options.output + "saved_params.json", 'r') as f:
            options.saved_params = json.load(f, encoding="latin-1")

        #options.saved_params = get_options_from_json(options.output + "saved_params.json")
    else:
        options.saved_params = dict()


    #parse what to do 
    get_condor = do_train = do_selection = do_fit = do_merge = False
    do_train = options.step == "train"
    get_condor = options.step == "get"

    get_condor = get_condor or do_selection

    total_batch_list = list(range(options.numBatches))

    kfold_options = []

    for k in range(options.kfolds):

        k_options = copy.deepcopy(options)
        k_list = copy.copy(total_batch_list)
        k_options.holdouts = list(range(k* ksize, (k+1) * ksize))
        
        for to_remove in k_options.holdouts:
            k_list.remove(to_remove)

        k_options.data_batch_list = k_list
        k_options.num_val_batch = len(k_list) // 4
        k_options.opts_list = []


        k_options.val_batch_list = k_options.data_batch_list[0 : k_options.num_val_batch]

        for j in k_options.val_batch_list: #validation batch range takes priority over regular batches
            if j in k_options.data_batch_list:
                k_options.data_batch_list.remove(j)

        #print(k, 'data', k_options.data_batch_list)
        #print(k, 'val', k_options.val_batch_list)

        kfold_options += [k_options]


    #Do trainings
    if(do_train):
        for k,k_options in enumerate(kfold_options):
            k_options.label += "_kfold%i" % k
            k_options.output += "kfold%i/" % k

            if(not os.path.exists(k_options.output)): os.system("mkdir %s" % k_options.output)

            to_run = []

            for idx, mbin in enumerate(mass_bin_idxs):
                outname = "%s/models/jrand_AE_kfold%i_mbin%i.h5" % (options.output, k, mbin)
                if(options.recover and os.path.exists(outname)): 
                    print("Skip jrand_AE_kfold%i_mbin%i.h5" % (k, mbin))
                    continue
                else:
                    print("Sub jrand_AE_kfold%i_mbin%i.h5" % (k, mbin))

                
                t_opts = copy.deepcopy(k_options)
                t_opts.mbin = mbin
                t_opts.max_events = 150000
                t_opts.num_epoch = 30
                t_opts.val_max_events = 30000
                t_opts.label = k_options.label + "_jrand_ae_mbin%i" % mbin
                t_opts.output = k_options.output + "jrand_ae_mbin%i/" % mbin
                if(not os.path.exists(t_opts.output)): os.system("mkdir %s" % t_opts.output)
                t_opts.step = "train"
                k_options.opts_list.append(t_opts.output + 'train_opts_%i.json' % idx)
                to_run.append(idx)

                train_ae_mbin(t_opts, idx, k, options.output)

            if(k_options.condor and len(to_run) > 0 ): #submit jobs for this kfold
                condor_dir = k_options.output + "condor_jobs/"
                if( not os.path.exists(condor_dir)): os.system("mkdir %s" % condor_dir)
                c_opts = condor_options().parse_args([])
                c_opts.nJobs = len(mass_bin_idxs)
                c_opts.outdir = condor_dir
                base_script = "../condor/scripts/train_from_json.sh"
                train_script = condor_dir + "train_script.sh"
                bb_name = k_options.fin.split("/")[-2]
                os.system("cp %s %s" % (base_script, train_script))
                os.system("sed -i 's/BB_NAME/%s/g' %s" % (bb_name, train_script))
                if(len(k_options.sig_file) > 0):
                    sig_fn = k_options.sig_file.split("/")[-1]
                    os.system("sed -i 's/SIGFILE/%s/g' %s" % ("--sig_file " + sig_fn, train_script))
                else:
                    os.system("sed -i 's/SIGFILE//g' %s" % (train_script))
                c_opts.script = train_script
                c_opts.name = k_options.label
                c_opts.input = k_options.opts_list
                c_opts.job_list = to_run
                c_opts.sub = True
                #c_opts.sub = False


                doCondor(c_opts)

                #for fname in k_options.opts_list:
                    #os.system("rm %s" % fname)

    if(get_condor):
        if(not os.path.exists(k_options.output)): os.system("mkdir %s/models" % options.output)
        for k,k_options in enumerate(kfold_options):
            k_options.label += "_kfold%i" % k

            for idx, mbin in enumerate(mass_bin_idxs):
                cmd = "xrdcp -p root://cmseos.fnal.gov//store/user/oamram/Condor_outputs/%s/model%i.h5 %s/models/jrand_AE_kfold%i_mbin%i.h5" % (k_options.label, idx, options.output, k, mbin)
                print(cmd)
                os.system(cmd)





if(__name__ == "__main__"):
    parser = input_options()
    parser.add_argument("--effs", nargs="+", default = [], type = float)
    parser.add_argument("--kfolds", default = 5, type = int)
    parser.add_argument("--lfolds", default = 4, type = int)
    parser.add_argument("--numBatches", default = 40, type = int)
    parser.add_argument("--step", default = "train",  help = 'Which step to perform (train, get, select, fit, roc, all)')
    parser.add_argument("--num_events", default = False, action = 'store_true', help = "Make limit plot in terms of num events (removes common prefactors)")
    parser.add_argument("--reload", action = 'store_true', help = "Reload based on previously saved options")
    parser.add_argument("--new", dest='reload', action = 'store_false', help = "Reload based on previously saved options")
    parser.add_argument("--recover", dest='recover', action = 'store_true', help = "Retrain jobs that failed")
    parser.set_defaults(reload=True)
    parser.add_argument("--condor", dest = 'condor', action = 'store_true')
    parser.add_argument("--no-condor", dest = 'condor', action = 'store_false')
    parser.set_defaults(condor=True)
    options = parser.parse_args()
    train_all_aes(options)
