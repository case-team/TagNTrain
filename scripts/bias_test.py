from full_run import *
from plotting.draw_sys_variations import *
import sys
from limit_set import *


def seed_opts(options, seed, sys = ""):

    t_opts = copy.deepcopy(options)
    t_opts.saved_params = None
    t_opts.BB_seed = seed

    label_str = '_%i' % seed

    t_opts.output = options.output + label_str  + "/"
    t_opts.label = options.label + "_" + label_str
    t_opts.redo_roc = False

    return t_opts

def get_sig_eff(outdir, eff = 1.0, sys = ""):
    fname = outdir + "fit_inputs_eff{eff}.h5".format(eff = eff)
    #eff_key = 'sig_eff_window' #with mjj window eff (for counting based limit)
    eff_key = 'sig_eff' #no mjj window (for shape based limit)
    if(len(sys) > 0): eff_key += "_" + sys
    if(not os.path.exists(fname)):
        print("Can't find fit inputs " + fname)
        return 0.

    with h5py.File(fname, "r") as f:
        sig_eff = f[eff_key][0]
        return sig_eff


def bias_test(options):
    if(len(options.label) == 0):
        if(options.output[-1] == "/"):
            options.label = options.output.split("/")[-2]
        else:
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
            rel_opts.sys_train_all = options.sys_train_all #quick fix
            if(abs(options.mjj_sig - 2500.) > 1.):  rel_opts.mjj_sig  = options.mjj_sig #quick fix
            rel_opts.step = options.step
            rel_opts.condor = options.condor
            rel_opts.seed = options.seed
            rel_opts.BB_seed = options.BB_seed
            rel_opts.refit = options.refit
            if(len(options.effs) >0): rel_opts.effs = options.effs
            if(options.sig_per_batch >= 0): rel_opts.sig_per_batch = options.sig_per_batch

            options = rel_opts

    seeds = range(options.num_toys)

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
    do_selection = options.step == "select"
    do_merge = "merge" in options.step
    do_fit = "fit" in options.step
    do_plot = options.step == "plot"
    do_roc = options.step == "roc"
    do_sys_train = options.step == "sys_train"
    do_sys_merge = options.step == "sys_merge"
    do_sys_selection = options.step == "sys_select"
    do_sys_plot = options.step == "sys_plot"
    do_clean = options.step == "clean"

    get_condor = get_condor or do_selection
    f_sig_effs = options.output + "sig_effs.npz"

    #Do trainings
    if(do_train):
        for seed in seeds:
            t_opts = seed_opts(options, seed)
            t_opts.step = "train"
            t_opts.condor = True
            full_run(t_opts)

    #Clean 
    if(do_clean):
        for seed in seeds:
            t_opts = seed_opts(options, seed)
            t_opts.step = "clean"
            full_run(t_opts)


    if(do_selection):
        if(len(options.effs) == 0):
            eff_point = mass_bin_select_effs[options.mbin]
            options.effs = [eff_point]
            print("Selecting with eff %.03f based on mbin %i" % (eff_point, options.mbin))

        for seed in seeds:
            t_opts = seed_opts(options, seed)
            t_opts.step = "select"
            t_opts.reload = True
            #t_opts.condor = options.condor
            t_opts.condor = True
            full_run(t_opts)


    if(do_merge):
        for seed in seeds:
            t_opts = seed_opts(options, seed)
            t_opts.step = "merge"
            t_opts.reload = True
            t_opts.condor = True
            full_run(t_opts)


    if(do_fit):
        for seed in seeds:
            t_opts = seed_opts(options, seed)
            t_opts.step = "fit"

            t_opts.reload = True
            t_opts.condor = True



            #run with specific shape
            #t_opts.fit_label = "sig_shape_"
            #t_opts.generic_sig_shape = False
            #full_run(t_opts)
            #os.system("mv %s %s" % (t_opts.output + "fit_results_%.1f.json" % options.mjj_sig, t_opts.output + "fit_results_sig_shape_%.1f.json" % options.mjj_sig))

            #run with generic shape
            if("Wp" in options.sig_file): 
                #max significance w/ generic shape comes from lower mass point b/c of long tail
                #TODO More proper is to scan over nearby and take lowest pval... 
                t_opts.mjj_sig = options.mjj_sig - 200.
            t_opts.fit_label = ""
            t_opts.generic_sig_shape = True
            full_run(t_opts)
            if("Wp" in options.sig_file): 
                os.system("mv %s %s" % (t_opts.output + "fit_results_%.1f.json" % (float(options.mjj_sig) -200.), t_opts.output + "fit_results_%.1f.json" % float(options.mjj_sig)))





    if(do_plot):
        if(not os.path.exists(os.path.join(options.output, "plots/"))):
            os.system("mkdir " + os.path.join(options.output, "plots/"))

        sig_effs = []
        excesses = []
        excess_uncs = []
        #get_signal_params(options)

        for seed in seeds:
        #for seed in range(10):
            t_opts = seed_opts(options, seed)
            sig_eff = float(get_sig_eff(t_opts.output, eff = options.effs[0]))

            fit_res = get_fit_results(outdir = t_opts.output,  m = options.mjj_sig)
            if(fit_res is not None): 
                obs_excess_events = float(fit_res.obs_excess_events)
                obs_excess_events_unc = float(fit_res.obs_excess_events_unc)
            else: 
                print("FAILED to get fit results for seed %i" % seed)
                obs_excess_events = -9999.


            sig_effs.append(sig_eff)
            excesses.append(obs_excess_events)
            excess_uncs.append(obs_excess_events_unc)


        mean_sig_eff = np.mean(sig_effs)
        std_sig_eff = np.std(sig_effs)
        expected_excess = options.numBatches * options.sig_per_batch * np.array(sig_effs)
        mean_expected_excess = np.mean(expected_excess)
        expected_variation = np.std(expected_excess)

        mean_excess = np.mean(excesses)
        err_mean_excess = np.std(excesses)/np.sqrt(len(excesses))


        print("Mean sig eff %.3f, std %.3f, expected excess %.1f +/- %.1f" % (mean_sig_eff, std_sig_eff, mean_expected_excess, expected_variation))
        print(excesses[:10])
        print("Results (mean, std): ", np.mean(excesses), np.std(excesses))

        fig = plt.figure(figsize=fig_size)
        num_bins = 20
        fontsize = 20

        ns, bins, patches = plt.hist(excesses, bins=num_bins, color='b', label="Toy Signal Yields", histtype='bar')

        plt.xlabel("Signal Yield", fontsize =fontsize)
        plt.tick_params(axis='x', labelsize=fontsize)
        plt.ylabel("nToys", fontsize =fontsize)
        plt.tick_params(axis='y', labelsize=fontsize)

        line_extra = 1.25

        plt.plot([mean_expected_excess]*2, [0, np.amax(ns)*line_extra], linestyle = "-", color = "black", linewidth = 3, label = "Mean Expected Yield (%.1f Events)" % mean_expected_excess)
        plt.plot([mean_expected_excess + expected_variation]*2, [0, np.amax(ns)*line_extra], linestyle = "--", color = "black", linewidth = 3, label = "+/- 1 Std Expected Yield")
        plt.plot([mean_expected_excess - expected_variation]*2, [0, np.amax(ns)*line_extra], linestyle = "--", color = "black", linewidth = 3,)

        plt.plot([mean_excess]*2, [0, np.amax(ns)*line_extra], linestyle = "-", color = "skyblue", linewidth = 3, label = "Mean Toy Yield (%.1f Events)" % mean_excess)
        plt.plot([mean_excess + err_mean_excess]*2, [0, np.amax(ns)*line_extra], linestyle = "--", color = "skyblue", linewidth = 3, label = "+/- 1-sigma Error on Mean Toy Yield")
        plt.plot([mean_excess - err_mean_excess]*2, [0, np.amax(ns)*line_extra], linestyle = "--", color = "skyblue", linewidth = 3,)

        plt.ylim((0, 1.7* np.amax(ns)))

        handles, labels = plt.gca().get_legend_handles_labels()
        order = [4,2,3,0,1]

        plt.legend( [handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper left', fontsize = 16)
        fname = os.path.join(options.output, "plots/injection_yields.png")
        print("Creating " + fname)
        plt.savefig(fname)

        fig = plt.figure(figsize=fig_size)

        #pulls
        eps = 1e-6
        pulls = ( np.array(excesses) - expected_excess)/(np.array(excess_uncs ) + eps)
        pulls = pulls[np.abs(pulls) < 10]

        ns, bins, patches = plt.hist(pulls, bins=num_bins, color='gray', label="Toys", histtype='bar')

        plt.xlabel("Pull of Signal Yield", fontsize =fontsize)
        plt.tick_params(axis='x', labelsize=fontsize)
        plt.ylabel("nToys", fontsize =fontsize)
        plt.tick_params(axis='y', labelsize=fontsize)

        mean_pull = np.mean(pulls)
        err_mean_pull = np.std(pulls)/np.sqrt(len(pulls))

        plt.plot([mean_pull]*2, [0, np.amax(ns)*line_extra], linestyle = "-", color = "skyblue", linewidth = 3, label = "Mean Toy Pull (%.1f +/- %.1f)" % (mean_pull, err_mean_pull))
        plt.plot([mean_pull + err_mean_pull]*2, [0, np.amax(ns)*line_extra], linestyle = "--", color = "skyblue", linewidth = 3, label = "+/- 1-sigma Error on Mean Toy Pull")
        plt.plot([mean_pull - err_mean_pull]*2, [0, np.amax(ns)*line_extra], linestyle = "--", color = "skyblue", linewidth = 3,)

        plt.ylim((0, 1.7* np.amax(ns)))
        plt.legend(loc='upper left', fontsize = 16)

        fname = os.path.join(options.output, "plots/injection_pulls.png")
        print("Creating " + fname)
        plt.savefig(fname)





    write_params(options.output + "saved_params.json", options.saved_params)




if(__name__ == "__main__"):
    parser = input_options()
    parser.add_argument("--sig_norm_unc", default = -1.0, type = float, help = "parameter for fit (uncertainty on signal efficiency)")
    parser.add_argument("--ae_dir", default = "", help = "directory with all the autoencoders (auto pick the right mbin and kfold)")
    parser.add_argument("--spbs", nargs="+", default = [], type = float)
    parser.add_argument("--effs", nargs="+", default = [], type = float)
    parser.add_argument("--kfolds", default = 5, type = int)
    parser.add_argument("--num_toys", default = 100, type = int)
    parser.add_argument("--lfolds", default = 4, type = int)
    parser.add_argument("--numBatches", default = 40, type = int)
    parser.add_argument("--do_TNT",  default=False, action = 'store_true',  help="Use TNT (default cwola)")
    parser.add_argument("--step", default = "train",  help = 'Which step to perform (train, get, select, fit, roc, all)')
    parser.add_argument("--counting_fit", default = False,  action = 'store_true', help = 'Do counting version of dijet fit')
    parser.add_argument("--num_events", default = False, action = 'store_true', help = "Make limit plot in terms of num events (removes common prefactors)")
    parser.add_argument("--sys_train_all", default = False, action = 'store_true', help = "Perform re-training for all systematics")
    parser.add_argument("--reload", action = 'store_true', help = "Reload based on previously saved options")
    parser.add_argument("--new", dest='reload', action = 'store_false', help = "Reload based on previously saved options")
    parser.set_defaults(reload=True)
    parser.set_defaults(deta=1.3)
    parser.add_argument("--condor", dest = 'condor', action = 'store_true')
    parser.add_argument("--no-condor", dest = 'condor', action = 'store_false')
    parser.add_argument("--recover", default = False, dest='recover', action = 'store_true', help = "Retrain jobs that failed")
    parser.add_argument("--refit", action = 'store_true', help = 'redo no selection signal fit')
    parser.set_defaults(condor=True)
    parser.set_defaults(num_models=3)
    options = parser.parse_args()
    bias_test(options)
