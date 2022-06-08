
from full_run import *

from matplotlib import pyplot as plt
import mplhep as hep
from os.path import join
from scipy.special import erf
import matplotlib


def plot_significances(input_files, out_dir):

    # build arrays of variables to plot
    mass_list = []
    pval_list = []
    signif_list = []

    for f in input_files:
        with open(f) as json_file:
            results = json.load(json_file)

        mass_list.append(results["mass"])
        pval_list.append(results["pval"])
        signif_list.append(results["signif"])

    xmax = np.amax(mass_list)

    corr_sigma = []
    for i in range(1, 6):
        tmp = 0.5-(0.5*(1+erf(i/np.sqrt(2)))-0.5*(1+erf(0/np.sqrt(2))))
        if i == 1:
            print(f"This should be 0.1586553: {tmp}")
        corr_sigma.append(tmp)
    mass_list = np.array(mass_list)
    pval_list = np.array(pval_list)
    signif_list = np.array(signif_list)

    plt.style.use(hep.style.CMS)
    plt.errorbar(mass_list, pval_list, fmt="ko")
    for i in range(len(corr_sigma)):
        xdash = np.concatenate(([0], mass_list, [8000]))
        plt.plot(xdash, np.full_like(xdash, corr_sigma[i]),
                 color="red", linestyle="dashed", linewidth=0.7)
        plt.text(xmax, corr_sigma[i]*1.25,
                 r"{} $\sigma$".format(i+1), fontsize=10,
                 color="red")
    plt.xlabel(r"$m_{jj}$ [GeV]")
    plt.ylabel("p-value")
    plt.yscale("log")
    plt.ylim(1e-7, 10)
    ax = plt.gca()
    ax.set_yticks([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1])
    y_minor = matplotlib.ticker.LogLocator(base=10.0,
                                           subs=np.arange(1.0, 10.0)*0.1,
                                           numticks=10)
    ax.yaxis.set_minor_locator(y_minor)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    plt.xlim(1400, xmax + 150.)
    hep.cms.text("Preliminary")
    plt.savefig(join(out_dir, "pval_plot.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.style.use(hep.style.CMS)
    hep.cms.text("Preliminary")
    plt.errorbar(mass_list, signif_list, fmt="ko")
    plt.xlabel(r"$m_{jj}$ [GeV]")
    plt.ylabel(r"Significance [$\sigma$]")
    plt.ylim(-0.2, None)
    plt.xlim(1400, xmax + 150.)
    plt.savefig(join(out_dir, "signif_plot.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Done!")





def mbin_opts(options, mbin, sys = ""):

    t_opts = copy.deepcopy(options)
    t_opts.mbin = mbin
    compute_mjj_window(t_opts)

    if(len(sys) > 0):
        label_str = sys
        if("rand" not in sys): t_opts.sig_sys = sys

    else:
        label_str = "mbin" +  str(mbin)
    t_opts.output = options.output + label_str  + "/"
    t_opts.label = options.label + "_" + label_str
    t_opts.redo_roc = False

    return t_opts



def full_scan(options):
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
            rel_opts.sys_train_all = options.sys_train_all #quick fix
            if(abs(options.mjj_sig - 2500.) > 1.):  rel_opts.mjj_sig  = options.mjj_sig #quick fix
            rel_opts.step = options.step
            if(len(options.effs) >0): rel_opts.effs = options.effs
            if(options.sig_per_batch >= 0): rel_opts.sig_per_batch = options.sig_per_batch

            options = rel_opts


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
    do_merge = options.step == "merge"
    do_fit = options.step == "fit"
    do_plot = options.step == "plot"

    get_condor = get_condor or do_selection

    #Do trainings
    if(do_train):
        for mbin in mass_bin_idxs:
            t_opts = mbin_opts(options, mbin)
            t_opts.step = "train"
            full_run(t_opts)

    if(do_selection):
        for mbin in mass_bin_idxs:
            t_opts = mbin_opts(options, mbin)
            t_opts.step = "select"
            t_opts.reload = True
            full_run(t_opts)

    if(do_merge):
        for mbin in mass_bin_idxs:
            t_opts = mbin_opts(options, mbin)
            t_opts.step = "merge"
            t_opts.reload = True
            t_opts.condor = True
            full_run(t_opts)



    if(do_fit):
        for mbin in mass_bin_idxs:
            t_opts = mbin_opts(options, mbin)
            t_opts.step = "fit"
            t_opts.reload = True
            t_opts.condor = False
            if(options.sideband): t_opts.fit_start = 2000.
            for sig_mass in mass_bin_sig_mass_map[mbin]:
                if(options.sideband and sig_mass < 2200.): continue
                t_opts.mjj_sig = sig_mass
                print("mbin %i, sig_mass %.0f" %(mbin, sig_mass))
                full_run(t_opts)


    if(do_plot):
        os.system("mkdir -p " + options.output + "plots/")
        sig_masses = []
        signifs = []
        pvals = []
        file_list = []
        for mbin in mass_bin_idxs:
            t_opts = mbin_opts(options, mbin)
            fit_plot = t_opts.output + 'sbFit_test_raw.png'
            for sig_mass in mass_bin_sig_mass_map[mbin]:
                if(options.sideband and sig_mass < 2200): continue

                os.system('cp ' + fit_plot + ' %s/plots/sbfit_mbin%i_mjj%.0f.png' % (options.output, mbin, sig_mass))

                fit_file = t_opts.output + 'fit_results_%.1f.json' % sig_mass
                if(os.path.exists(fit_file)):
                    file_list.append(fit_file)
                    with open(fit_file, 'r') as f:
                        fit_params = json.load(f, encoding="latin-1")
                        sig_masses.append(fit_params['mass'])
                        signifs.append(fit_params['signif'])
                        pvals.append(fit_params['pval'])
                else:
                    print("Missing %s" % fit_file)
        
        print(list(zip(sig_masses, signifs)))
        plot_significances(file_list, options.output + "plots/")
            

    write_params(options.output + "saved_params.json", options.saved_params)




if(__name__ == "__main__"):
    parser = input_options()
    parser.add_argument("--fit_start", default = -1.,  type = float, help = 'Lowest mjj value for dijet fit')
    parser.add_argument("--sig_norm_unc", default = -1.0, type = float, help = "parameter for fit (uncertainty on signal efficiency)")
    parser.add_argument("--ae_dir", default = "", help = "directory with all the autoencoders (auto pick the right mbin and kfold)")
    parser.add_argument("--effs", nargs="+", default = [], type = float)
    parser.add_argument("--kfolds", default = 5, type = int)
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
    parser.add_argument("--condor", dest = 'condor', action = 'store_true')
    parser.add_argument("--no-condor", dest = 'condor', action = 'store_false')
    parser.set_defaults(condor=True)
    options = parser.parse_args()
    full_scan(options)
