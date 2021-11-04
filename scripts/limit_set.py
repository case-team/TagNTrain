from full_run import *

def spb_opts(options, spb):

    t_opts = copy.deepcopy(options)
    t_opts.sig_per_batch = spb
    t_opts.output = options.output + "spb" +  str(spb) + "/"
    t_opts.label = options.label + "_spb" +  str(spb)

    return t_opts


def make_limit_plot(options, sig_effs):
    fout = options.output + options.label + "_limit_plot.png"

    lumi = 28.
    n_evts_exc = 80.
    n_evts_exc_no_sel = 800.

    if(options.sig_idx ==1):
        preselection_eff = 0.92 * 0.88
        sig_eff_no_cut = 0.85
    elif(options.sig_idx == 3):
        preselection_eff = 0.76*0.59
        sig_eff_no_cut =  0.5
    else:
        print("Sig preselection stuff not computed for sig %i" % options.sig_idx)
        sys.exit(1)



    injected_xsecs = np.array([( spb*options.numBatches / lumi / preselection_eff) for spb in options.spbs])
    excluded_xsecs = np.array([(n_evts_exc / (sig_eff * preselection_eff * lumi)) for sig_eff in sig_effs])
    no_sel_limit = n_evts_exc_no_sel / (preselection_eff * sig_eff_no_cut * lumi)
    print("Limit without NN selection: %.1f" % no_sel_limit)



    best_lim = 999999
    for i,s_inj in enumerate(injected_xsecs):
        s_excl = excluded_xsecs[i]
        lim = max(s_inj, s_excl)
        if(lim < best_lim):
            best_lim = lim
            best_i = i

        print(options.spbs[i], s_inj, s_excl)

    print("Best lim : %.3f"  % best_lim)
        
    fig_size = (12,9)
    plt.figure(figsize=fig_size)
    size = 0.4

    x_stop = max(injected_xsecs)
    xline = np.arange(0,x_stop,x_stop/10)
    plt.plot(xline, xline, linestyle = "--", color = "black", linewidth = 2, label = "Excluded = Injected")
    if(no_sel_limit >0):
        plt.plot(xline, [no_sel_limit]*10, linestyle = "--", color = "green", linewidth = 2, label = "Inclusive Limit (%.1f fb)" % no_sel_limit)

    vertical_line = injected_xsecs[best_i] > excluded_xsecs[best_i]
    lim_line_max = min(injected_xsecs[best_i], excluded_xsecs[best_i])*0.97

    best_lim_label = "Best Limit (%.1f fb)" % best_lim
    if(vertical_line):
        plt.plot([best_lim, best_lim], [0., lim_line_max], linestyle="--", color = "red", linewidth =2, label = best_lim_label)
    else: #horizontal line
        plt.plot( [0., lim_line_max], [best_lim, best_lim], linestyle="--", color = "red", linewidth =2, label = best_lim_label)

    plt.scatter(injected_xsecs, excluded_xsecs, c ='b' , s=40.0, label="Injections")

    off_plots = excluded_xsecs > (x_stop * 1.2)

    num_off = np.sum(off_plots)

    off_ys = [x_stop * 1.18] * num_off

    plt.scatter(injected_xsecs[off_plots], off_ys, c = 'b', s = 40.0, marker ="^")


    plt.ylim([0., x_stop * 1.2])
    plt.xlim([0., x_stop * 1.2])

    plt.xlabel("Injected cross section (fb)", fontsize=20)
    plt.ylabel(" 'Excluded' cross section (fb)", fontsize=20)
    plt.tick_params(axis='y', labelsize=16)
    plt.tick_params(axis='x', labelsize=16)

    fs_leg = 16
    plt.legend(loc="upper right", fontsize = fs_leg)
    print("saving %s" % fout)
    plt.savefig(fout)


def get_sig_eff(outdir):
    fname = outdir + "fit_inputs.h5"
    eff_key = 'sig_eff_window'
    if(not os.path.exists(fname)):
        print("Can't find fit inputs " + fname)
        return 0.

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
            print(rel_opts.__dict__)
            rel_opts.keep_LSF = options.keep_LSF #quick fix
            rel_opts.step = options.step
            if(len(options.effs) >0): rel_opts.effs = options.effs
            if(options.sig_per_batch >= 0): rel_opts.sig_per_batch = options.sig_per_batch
            if(len(options.spbs) > 0):
                spbs_to_run = options.spbs
                cur_spbs = rel_opts.spbs
                for spb in options.spbs:
                    if(spb not in cur_spbs):
                        cur_spbs.append(spb)
                rel_opts.spbs = sorted(cur_spbs)
            else:
                spbs_to_run = rel_opts.spbs

            options = rel_opts
    else:
        spbs_to_run = options.spbs


    #save options
    options_dict = options.__dict__
    write_options_to_pkl(options_dict, options.output + "run_opts.pkl" )

    #parse what to do 
    get_condor = do_train = do_selection = do_fit = do_merge = False
    do_train = options.step == "train"
    get_condor = options.step == "get"
    do_selection = options.step == "select"
    do_merge = options.step == "merge"
    do_fit = options.step == "fit"
    do_plot = options.step == "plot"

    get_condor = get_condor or do_selection
    f_sig_effs = options.output + "sig_effs.npz"
    print(options.spbs)
    print("To run", spbs_to_run)

    #Do trainings
    if(do_train):
        for spb in spbs_to_run:
            t_opts = spb_opts(options, spb)
            t_opts.step = "train"
            t_opts.condor = True
            full_run(t_opts)

    if(do_selection):
        for spb in spbs_to_run:
            t_opts = spb_opts(options, spb)
            t_opts.step = "select"
            t_opts.reload = True
            t_opts.condor = True
            full_run(t_opts)

    if(do_merge):
        for spb in options.spbs:
            t_opts = spb_opts(options, spb)
            t_opts.step = "merge"
            t_opts.reload = True
            t_opts.condor = True
            full_run(t_opts)

    if(do_plot):
        sig_effs = []
        for spb in options.spbs:
            sig_eff = get_sig_eff(options.output + "spb%i/" % spb)
            sig_effs.append(sig_eff)

        print(sig_effs)
        sig_effs = np.array(sig_effs)
        np.savez(f_sig_effs, sig_eff = sig_effs)


        make_limit_plot(options, sig_effs)




if(__name__ == "__main__"):
    parser = input_options()
    parser.add_argument("--spbs", nargs="+", default = [], type = int)
    parser.add_argument("--effs", nargs="+", default = [], type = float)
    parser.add_argument("--kfolds", default = 5, type = int)
    parser.add_argument("--lfolds", default = 4, type = int)
    parser.add_argument("--numBatches", default = 40, type = int)
    parser.add_argument("--do_TNT",  default=False, action = 'store_true',  help="Use TNT (default cwola)")
    parser.add_argument("--step", default = "train",  help = 'Which step to perform (train, get, select, fit, roc, all)')
    parser.add_argument("--counting_fit", default = False,  action = 'store_true', help = 'Do counting version of dijet fit')
    parser.add_argument("--reload", default = False, action = 'store_true', help = "Reload based on previously saved options")
    options = parser.parse_args()
    limit_set(options)
