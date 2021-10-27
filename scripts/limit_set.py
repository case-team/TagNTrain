from full_run import *

def spb_opts(options, spb):

    t_opts = copy.deepcopy(options)
    t_opts.sig_per_batch = spb
    t_opts.output = options.output + "spb" +  str(spb) + "/"
    t_opts.label = options.label + "_spb" +  str(spb)

    return t_opts


def make_limit_plot(s_injs, s_excls, fout):
    best_lim = 999999
    for i,s_inj in enumerate(s_injs):
        s_excl = s_excls[i]
        lim = max(s_inj, s_excl)
        best_lim = min(excl, best_lim)

        print(s_inj, s_excl, lim)
    print("Best lim : %.3f", best_lim)


def get_sig_eff(outdir):
    fname = outdir + "fit_inputs.h5"
    eff_key = 'sig_eff_window'

    with h5py.File(fname, "r") as f:
        sig_eff = f[eff_key][0]
        return sig_eff

def limit_set(options):

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
            if(options.sig_per_batch >= 0): rel_opts.sig_per_batch = options.sig_per_batch
            options = rel_opts
    else:
        #save options
        options_dict = options.__dict__
        write_options_to_pkl(options_dict, options.output + "run_opts.pkl" )

    #parse what to do 
    get_condor = do_train = do_selection = do_fit = False
    do_train = options.step == "train"
    get_condor = options.step == "get"
    do_selection = options.step == "select"
    do_fit = options.step == "fit"
    do_plot = options.step == "plot"

    get_condor = get_condor | (do_selection and options.condor)
    f_sig_effs = options.output + "sig_effs.npz"
    print(options.spbs)

    #Do trainings
    if(do_train):
        for spb in options.spbs:
            t_opts = spb_opts(options, spb)
            t_opts.step = "train"
            t_opts.condor = True
            full_run(t_opts)

    if(do_selection):
        sig_effs = []
        for spb in options.spbs:
            t_opts = spb_opts(options, spb)
            t_opts.step = "select"
            t_opts.reload = True
            full_run(t_opts)
            sig_eff = get_sig_eff(t_opts.output)
            sig_effs.append(sig_eff)
        sig_effs = np.array(sig_effs)
        np.savez(f_sig_effs, sig_eff = sig_effs)

    if(do_plot):
        lumi = 28.
        n_evts_exc = 1000.
        preselection_eff = 0.8

        f_np = np.load(f_sig_effs)
        sig_effs = f_np['sig_eff']
        injected_xsecs = [( spb*options.numBatches / lumi / preselection_eff) for spb in options.spbs]
        excluded_xsecs = [(n_evts_exc * sig_eff * preselection_eff / lumi) for sig_eff in sig_effs]

        make_limit_plot(injected_xsecs, excluded_xsecs, options.output + "limit_plot.png")




if(__name__ == "__main__"):
    parser = input_options()
    parser.add_argument("--spbs", nargs="+", default = [], type = int)
    parser.add_argument("--effs", nargs="+", default = [], type = float)
    parser.add_argument("--kfolds", default = 5, type = int)
    parser.add_argument("--lfolds", default = 4, type = int)
    parser.add_argument("--numBatches", default = 40, type = int)
    parser.add_argument("--do_TNT",  default=False, action = 'store_true',  help="Use TNT (default cwola)")
    parser.add_argument("--condor", default = True)
    parser.add_argument("--step", default = "train",  help = 'Which step to perform (train, get, select, fit, roc, all)')
    parser.add_argument("--reload", default = False, action = 'store_true', help = "Reload based on previously saved options")
    options = parser.parse_args()
    limit_set(options)
