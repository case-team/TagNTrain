from full_run import *
from plotting.draw_sys_variations import *
import sys
from limit_set import *


def seed_opts(options, seed, sys = "", flip = False):

    t_opts = copy.deepcopy(options)
    t_opts.saved_params = None
    t_opts.BB_seed = seed

    label_str = '_%i' % seed
    if(flip):
        label_str += "_flip"
        t_opts.sig_file = options.sig2_file
        t_opts.sig_per_batch = options.sig2_per_batch

        t_opts.sig2_file = options.sig_file
        t_opts.sig2_per_batch = options.sig_per_batch

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
        #print("Can't find fit inputs " + fname)
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
            rel_opts.plot_label = options.plot_label
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
    do_selection = 'select' in options.step
    do_merge = "merge" in options.step
    do_fit = "fit" in options.step
    do_eff = 'eff' in options.step
    do_plot = options.step == "plot"
    do_roc = options.step == "roc"
    do_sys_train = options.step == "sys_train"
    do_sys_merge = options.step == "sys_merge"
    do_sys_selection = options.step == "sys_select"
    do_sys_plot = options.step == "sys_plot"
    do_clean = options.step == "clean"

    flip_signals = 'flip' in options.step

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

    if(get_condor):
        for seed in seeds:
            t_opts = seed_opts(options, seed)
            t_opts.step = "get"
            full_run(t_opts)

    if(do_selection):
        if(len(options.effs) == 0):
            eff_point = mass_bin_select_effs[options.mbin]
            options.effs = [eff_point]
            print("Selecting with eff %.03f based on mbin %i" % (eff_point, options.mbin))

        if(flip_signals):
            for seed in seeds:

                copy_opts = copy.deepcopy(options)
                copy_opts.output = options.output + "_%i_flip/" % seed
                copy_opts.orig_dir = options.output + "_%i/" % seed
                copy_taggers(copy_opts)

                t_opts = seed_opts(options, seed, flip = True)

                t_opts.step = "select"
                t_opts.reload = False

                #t_opts.condor = options.condor
                t_opts.condor = True
                full_run(t_opts)

        else:
            for seed in seeds:
                t_opts = seed_opts(options, seed)
                t_opts.step = "select"
                t_opts.reload = True
                #t_opts.condor = options.condor
                t_opts.condor = True
                full_run(t_opts)


    if(do_merge):
        for seed in seeds:
            t_opts = seed_opts(options, seed, flip = flip_signals)
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


    if(do_eff):
        sig_effs = []
        for seed in seeds:
            t_opts = seed_opts(options, seed, flip = flip_signals)
            eff = float(get_sig_eff(t_opts.output, eff = options.effs[0]))
            if(eff > 1e-6): sig_effs.append(eff)

        print("Effs", str(sig_effs))
        err_mean = np.std(sig_effs) / np.sqrt(len(sig_effs)-1)
        print("Mean Eff. %.3f +/- %.3f " % (np.mean(sig_effs), err_mean))
        print("Std Dev. Eff. %.3f" % np.std(sig_effs))




    if(do_plot):
        if(not os.path.exists(os.path.join(options.output, "plots/"))):
            os.system("mkdir " + os.path.join(options.output, "plots/"))

        sig_effs = []
        sig_effs_inj = []
        excesses = []
        excess_uncs = []
        obs_lims = []
        exp_lims = []

        meas_sig_xsecs = []
        err_sig_xsecs = []
        obs_lim_xsecs = []
        exp_lim_xsecs = []
        signifs = []

        preselection_eff,_,_ = get_preselection_params(options.sig_file, options.hadronic_only, 3000)
        lumi = 26.81
        def convert_to_xsec(nevts, sig_eff):
            return nevts / (sig_eff * preselection_eff * lumi)

        true_sig_xsec = convert_to_xsec(options.sig_per_batch * options.numBatches, 1.0)
        print('true injected xsec %.1f' % true_sig_xsec)

        injections = [
            "/uscms_data/d3/oamram/CASE_analysis/src/CASE/TagNTrain/runs/TNT_bias_test_XYY_3TeV_spb01_oct25/",
            "/uscms_data/d3/oamram/CASE_analysis/src/CASE/TagNTrain/runs/TNT_bias_test_XYY_3TeV_spb1_oct25/",
            "/uscms_data/d3/oamram/CASE_analysis/src/CASE/TagNTrain/runs/TNT_bias_test_XYY_3TeV_spb2_oct25/",
            "/uscms_data/d3/oamram/CASE_analysis/src/CASE/TagNTrain/runs/TNT_bias_test_XYY_spb3_may21/",
            "/uscms_data/d3/oamram/CASE_analysis/src/CASE/TagNTrain/runs/TNT_bias_test_XYY_3TeV_spb4_oct25/",
            #"/uscms_data/d3/oamram/CASE_analysis/src/CASE/TagNTrain/runs/TNT_bias_test_XYY_3TeV_spb5_oct25/",
            ]

        #injection_spbs = [0.1, 1,2,3,4,5]
        injection_spbs = [0.1, 1,2,3,4]

        sig_effs = [ [] for i in range(len(injections))]


        for seed in seeds:
            for i,inj_dir in enumerate(injections):
                loc = inj_dir + ("_%i/" % seed)
                eff = float(get_sig_eff(loc, eff = options.effs[0]))
                if(eff > 1e-6): sig_effs[i].append(eff)

            t_opts = seed_opts(options, seed)
            fit_res = get_fit_results(outdir = t_opts.output,  m = options.mjj_sig)
            if(fit_res is not None): 
                obs_excess_events = float(fit_res.obs_excess_events)
                obs_excess_events_unc = float(fit_res.obs_excess_events_unc)
                obs_lim_events = fit_res.obs_lim_events
                exp_lim_events = fit_res.exp_lim_events

                signif = float(fit_res.signif) if 'asimov_signif' not in fit_res.__dict__.keys() else float(fit_res.asimov_signif)

                excesses.append(obs_excess_events)
                excess_uncs.append(obs_excess_events_unc)
                obs_lims.append(obs_lim_events)
                exp_lims.append(exp_lim_events)
                signifs.append(signif)
            else: 
                print("FAILED to get fit results for seed %i" % seed)



        
        sig_eff_avgs = [np.mean(sig_effs[i]) for i in range(len(sig_effs))]
        avg_signif = np.mean(signifs)
        print("Average efficiencies: " + str(sig_eff_avgs))
        print("Average significance " + str(avg_signif))

        my_idx = injection_spbs.index(options.sig_per_batch)
        local_sig_effs = sig_effs[my_idx]



        def plot_hist(pulls, xlabel, fout = "", text = True):
            fig = plt.figure(figsize=fig_size)
            num_bins = 20
            fontsize = 24
            ns, bins, patches = plt.hist(pulls, bins=num_bins, color='gray', label="Toys", histtype='bar')

            plt.xlabel(xlabel, fontsize =fontsize)
            plt.tick_params(axis='x', labelsize=fontsize)
            plt.ylabel("nToys", fontsize =fontsize)
            plt.tick_params(axis='y', labelsize=fontsize)


            if(text):

                mean_pull = np.mean(pulls)
                std_pull = np.std(pulls)
                err_mean_pull =  std_pull /np.sqrt(len(pulls))
                err_std_pull =  std_pull /np.sqrt(2*len(pulls)-2)
                txt = "Mean = %.1f +/- %.1f" % (mean_pull, err_mean_pull)
                txt2 = "Std Dev = %.2f +/- %.2f" % (std_pull, err_std_pull)

                plt.text(0.3, 0.9, txt, transform = fig.axes[0].transAxes, fontsize = 18)
                plt.text(0.3, 0.85, txt2, transform = fig.axes[0].transAxes, fontsize = 18)

            plt.ylim((0, 1.7* np.amax(ns)))

            if(fout != ""):
                print("Creating " + fout)
                plt.savefig(fout)
            return fig


        #injection cross section used in limit setting (cross section you 'think' you injected)
        spb_injs = [(spb - options.sig_per_batch) for spb in injection_spbs if spb >= options.sig_per_batch]
        effs_filtered = sig_eff_avgs[-len(spb_injs):]

        #assume eff becomes flat at very high xsec
        effs_filtered.append(effs_filtered[-1])
        spb_injs.append(100)

        xsec_injs = [convert_to_xsec( spb_injs[i] * options.numBatches, 1.0) for i in range(len(spb_injs))]

        print(effs_filtered)
        print(xsec_injs)
        yields = [xsec_injs[i] * lumi *  effs_filtered[i] * preselection_eff for i in range(len(xsec_injs))]
        print(yields)
        print(obs_lims[:5])


        xsec_obs_lims = np.interp(obs_lims, yields, xsec_injs)
        xsec_exp_lims = np.interp(exp_lims, yields, xsec_injs)
        print(xsec_obs_lims[:10])
        print(xsec_exp_lims[:10])
        print(np.mean(xsec_obs_lims))


        fout = os.path.join(options.output, "plots/xsec_lims.png")
        fig = plot_hist(xsec_obs_lims, "95% Upper Limit on Signal Cross Section", fout = "", text = False)
        plt.vlines([true_sig_xsec], 0, 10, color = 'blue', linestyle = 'dashed')

        coverage = 100. * np.mean(xsec_obs_lims > true_sig_xsec)
        cov_str = "Coverage = %.1f %%" % coverage
        print(cov_str)
        inj_str = "True cross section in 'data' = %.1f fb, %.1f$\sigma$ avg. excess " % (true_sig_xsec, avg_signif)
        plt.text(0.5, 0.9, inj_str, transform = fig.axes[0].transAxes, horizontalalignment = 'center', fontsize = 20)
        plt.text(0.5, 0.8, cov_str, transform = fig.axes[0].transAxes, horizontalalignment = 'center', fontsize = 24)
        plt.savefig(fout)


        fout_map = options.output + "sig_eff_map.h5"
        f = h5py.File(fout_map, "w")
        f.create_dataset("injected_xsecs", data = xsec_injs[:-1])
        f.create_dataset("effs", data = effs_filtered[:-1])
        norm_factor = lumi * preselection_eff
        f.create_dataset("norm_factor", data = [norm_factor])
        print("\nOutputing sig eff map")
        print("Norm factor %.3f" % norm_factor)




    write_params(options.output + "saved_params.json", options.saved_params)




if(__name__ == "__main__"):
    parser = input_options()
    parser.add_argument("--sig_norm_unc", default = -1.0, type = float, help = "parameter for fit (uncertainty on signal efficiency)")
    parser.add_argument("--ae_dir", default = "", help = "directory with all the autoencoders (auto pick the right mbin and kfold)")
    parser.add_argument("--plot_label", default = "", help = "directory with all the autoencoders (auto pick the right mbin and kfold)")
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
