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
            if(seed <= 44): continue
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
        sig_effs_inj = []
        excesses = []
        excess_uncs = []
        obs_lims = []
        exp_lims = []

        meas_sig_xsecs = []
        err_sig_xsecs = []
        obs_lim_xsecs = []
        exp_lim_xsecs = []

        preselection_eff,_,_ = get_preselection_params(options.sig_file, options.hadronic_only, 3000)
        lumi = 26.81
        def convert_to_xsec(nevts, sig_eff):
            return nevts / (sig_eff * preselection_eff * lumi)

        true_sig_xsec = convert_to_xsec(options.sig_per_batch * options.numBatches, 1.0)
        print('true injected xsec %.1f' % true_sig_xsec)

        other_injection = "/uscms_data/d3/oamram/CASE_analysis/src/CASE/TagNTrain/runs/TNT_bias_test_XYY_spb3_may21/"
        inj_spb = 3
        #other_injection = "/uscms_data/d3/oamram/CASE_analysis/src/CASE/TagNTrain/runs/TNT_bias_test_XYY_3TeV_spb2_oct25/"
        #inj_spb = 2

        for seed in seeds:
        #for seed in range(10):
            if(seed == 44): continue
            t_opts = seed_opts(options, seed)
            sig_eff = float(get_sig_eff(t_opts.output, eff = options.effs[0]))

            alt_injection = other_injection + ("_%i/" % seed)
            sig_eff_inj = float(get_sig_eff(alt_injection, eff = options.effs[0]))

            fit_res = get_fit_results(outdir = t_opts.output,  m = options.mjj_sig)
            if(fit_res is not None): 
                obs_excess_events = float(fit_res.obs_excess_events)
                obs_excess_events_unc = float(fit_res.obs_excess_events_unc)
                obs_lim_events = fit_res.obs_lim_events
                exp_lim_events = fit_res.exp_lim_events
            else: 
                print("FAILED to get fit results for seed %i" % seed)
                obs_excess_events = -9999.


            sig_effs.append(sig_eff)
            sig_effs_inj.append(sig_eff_inj)
            excesses.append(obs_excess_events)
            excess_uncs.append(obs_excess_events_unc)
            obs_lims.append(obs_lim_events)
            exp_lims.append(exp_lim_events)

            meas_sig_xsec = convert_to_xsec(obs_excess_events, sig_eff)
            err_sig_xsec = (obs_excess_events_unc /obs_excess_events) * meas_sig_xsec
            obs_lim_xsec = convert_to_xsec(obs_lim_events, sig_eff)
            exp_lim_xsec = convert_to_xsec(exp_lim_events, sig_eff)

            meas_sig_xsecs.append(meas_sig_xsec)
            err_sig_xsecs.append(err_sig_xsec)
            obs_lim_xsecs.append(obs_lim_xsecs)
            exp_lim_xsecs.append(exp_lim_xsecs)




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


        def plot_hist(pulls, xlabel, fout = "", text = True):
            fig = plt.figure(figsize=fig_size)
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
            plt.legend(loc='upper left', fontsize = 16)

            if(fout != ""):
                print("Creating " + fout)
                plt.savefig(fout)
            return fig


        #signal yield pulls
        eps = 1e-6
        pulls_sig_yield = ( np.array(excesses) - expected_excess)/(np.array(excess_uncs ) + eps)
        pulls_sig_yield = pulls_sig_yield[np.abs(pulls_sig_yield) < 10]
        fout = os.path.join(options.output, "plots/injection_pulls.png")
        plot_hist(pulls_sig_yield, "Pull of Signal Yield", fout)


        #xsec pulls v1
        fig = plt.figure(figsize=fig_size)
        eps = 1e-6
        pull_sig_xsec = [(meas_sig_xsecs[i] - true_sig_xsec) / (err_sig_xsecs[i] + eps) for i in range(len(meas_sig_xsecs)) if err_sig_xsecs[i] > eps] 
        print(meas_sig_xsecs[:5], err_sig_xsecs[:5], pull_sig_xsec[:5])

        fout = os.path.join(options.output, "plots/injection_pulls_xsec.png")
        plot_hist(pull_sig_xsec, "Pull of Signal Cross Section", fout)

        #xsec pulls v2
        xsec_pulls_sampled = []
        xsec_sampled = []
        xsec_obs_lims = []
        xsec_obs_lims_inj = []
        nToys = len(sig_effs)
        sig_effs =np.array(sig_effs)
        sig_effs_inj = np.array(sig_effs_inj)
        #sig_effs = sig_effs[(sig_effs > eps)  & (sig_effs < 1.0)]

        #injection cross section used in limit setting (cross section you 'think' you injected)
        xsec_inj = convert_to_xsec((inj_spb - options.sig_per_batch) * options.numBatches, 1.0)
        print("Lim set inj xsec %.2f" % xsec_inj)

        rng = np.random
        rng.seed(123)
        for i in range(nToys):
            #other_idxs = np.delete(np.arange(0,nToys), i)
            other_idxs = np.arange(0,nToys)
            selected = rng.choice(other_idxs, 5, replace = False)

            sig_eff_inj =  np.mean(sig_effs_inj[selected])

            sig_eff_mean = np.mean(sig_effs[selected])
            sig_eff_std = np.std(sig_effs[selected])

            meas_sig_xsec_sampled = convert_to_xsec(excesses[i], sig_eff_mean)
            err_sig_eff = (sig_eff_std / sig_eff_mean)
            err_tot = meas_sig_xsec_sampled * (err_sig_eff**2 + (err_sig_xsecs[i]/meas_sig_xsecs[i])**2) ** (0.5)

            xsec_sampled.append(meas_sig_xsec_sampled)
            xsec_pulls_sampled.append((meas_sig_xsec_sampled  - true_sig_xsec) / err_tot)

            xsec_obs_lims.append(convert_to_xsec(obs_lims[i], sig_effs[i]))

            xsec_lim_inj = convert_to_xsec(obs_lims[i], sig_eff_inj)
            xsec_lim_inj = max(xsec_inj, xsec_lim_inj)
            xsec_obs_lims_inj.append(xsec_lim_inj)



        fout = os.path.join(options.output, "plots/injection_pulls_xsec_sampled.png")
        plot_hist(xsec_pulls_sampled, "Pull of Signal Cross Section", fout = "")
        plt.xlim(-3,3)
        plt.savefig(fout)

        fout = os.path.join(options.output, "plots/injection_xsec_sampled.png")
        plot_hist(xsec_sampled, "Observed Signal Cross Section", fout = "")
        plt.vlines([true_sig_xsec], 0, 15, color = 'blue', linestyle = 'dashed')
        plt.savefig(fout)

        fout = os.path.join(options.output, "plots/xsec_lims.png")
        fig = plot_hist(xsec_obs_lims, "95% Upper Limit on Signal Cross Section", fout = "", text = False)
        plt.vlines([true_sig_xsec], 0, 10, color = 'blue', linestyle = 'dashed')

        xsec_obs_lims = np.array(xsec_obs_lims)
        coverage = 100. * np.mean(xsec_obs_lims > true_sig_xsec)
        plt.text(0.3, 0.9, "Coverage = %.1f %%" % coverage, transform = fig.axes[0].transAxes, fontsize = 18)
        plt.savefig(fout)

        fout = os.path.join(options.output, "plots/xsec_lims_inj.png")
        fig = plot_hist(xsec_obs_lims_inj, "95% Upper Limit on Signal Cross Section", fout = "", text = False)
        plt.vlines([true_sig_xsec], 0, 10, color = 'blue', linestyle = 'dashed')

        xsec_obs_lims_inj = np.array(xsec_obs_lims_inj)
        coverage = 100. * np.mean(xsec_obs_lims_inj > true_sig_xsec)
        plt.text(0.3, 0.9, "Coverage = %.1f %%" % coverage, transform = fig.axes[0].transAxes, fontsize = 18)
        plt.text(0.3, 0.85, options.plot_label, transform = fig.axes[0].transAxes, fontsize = 18)
        plt.savefig(fout)

        print("Avg data sig eff %.3f" % np.mean(sig_effs))
        print("Avg data inj sig eff %.3f" % np.mean(sig_effs_inj))
        print("Avg taggerData lim %.3f" % np.mean(xsec_obs_lims))
        print("Avg anomaly lim %.3f" % np.mean(xsec_obs_lims_inj))




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
