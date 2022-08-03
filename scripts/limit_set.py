from full_run import *
from plotting.draw_sys_variations import *

dedicated_searches = {
        "XToHY_MX_2600_MY_200_UL17_TIMBER.h5" : (1, "B2G-21-003"),
        }

fit_results = {
        2000: (210., 2260.),
        2500: (93., 1000.),
        3000: (45., 540.),
        3500: (21., 280.),
        4000: (11., 160.),
        4500: (6., 92.),
        5000: (4., 53.),
        }


def spb_opts(options, spb, sys = ""):
    spb = float(spb)

    t_opts = copy.deepcopy(options)
    t_opts.sig_per_batch = spb

    if(len(sys) > 0):
        label_str = sys
        if("rand" not in sys): t_opts.sig_sys = sys

    else:
        label_str = "spb" +  str(spb)
    t_opts.output = options.output + label_str  + "/"
    t_opts.label = options.label + "_" + label_str
    t_opts.redo_roc = False

    return t_opts

def get_optimal_spb(options, sig_effs):

    if(int(options.mjj_sig) in fit_results.keys()):
        n_evts_exc, n_evts_exc_no_sel = fit_results[int(options.mjj_sig)]

    else:
        print("Can't find signal mjj (%i) in fit results dict" % int(options.mjj))
        n_evts_exc = 93.
        n_evts_exc_no_sel = 1000.
    print("N_evts_exc %.0f No cut %.0f " % ( n_evts_exc, n_evts_exc_no_sel))

    get_signal_params(options)

    sig_effs = np.clip(sig_effs, 1e-4, 1.0)
    injected_xsecs = np.array([( spb*options.numBatches / options.lumi / options.preselection_eff / options.hadronic_only_eff) for spb in options.spbs])
    excluded_xsecs = np.array([(n_evts_exc / (options.hadronic_only_eff * sig_eff * options.preselection_eff * options.lumi)) for sig_eff in sig_effs])

    lims = [ max(injected_xsecs[i], excluded_xsecs[i]) for i in range(len(options.spbs))]
    print("xsec lims are : ", lims)
    optimal_spb = options.spbs[np.argmin(lims)]
    print("Optimal spb %i" % optimal_spb)
    return optimal_spb

def get_preselection_params(sig_fname):
    print("Getting preselection params from %s" % sig_fname)
    if(not os.path.exists(sig_fname)):
        print("Can't find file " + sig_fname)
        return 0., 1.0

    f = h5py.File(sig_fname, "r")
    presel_eff = f['preselection_eff'][0]
    deta_eff = f['d_eta_eff'][0]
    #print(deta_eff)
    is_lep = f['event_info'][:,4]
    hadronic_only = 1.0 - np.mean(is_lep)
    return presel_eff * deta_eff, is_lep

def get_signal_params(options):
    #modifies options struct
    options.lumi = 28.
    options.dedicated_lim = -1
    options.dedicated_label = ""
    if(len(options.sig_file) > 0):

        options.preselection_eff, hadronic_only_ = get_preselection_params(options.sig_file)

        
        if(options.hadronic_only): options.hadronic_only_eff = hadronic_only_
        else: options.hadronic_only_eff = 1.0
        sig_fn = options.sig_file.split('/')[-1]
        if( sig_fn in dedicated_searches.keys()):
            options.dedicated_lim, AN_name = dedicated_searches[sig_fn]
        
            #lumi scale the limit
            options.dedicated_lim *= (138./options.lumi)**0.5
            options.dedicated_label = "Dedicated Search, %s (%.1f fb)" % (AN_name, options.dedicated_lim)
    elif(options.sig_idx ==1):
        options.preselection_eff = 0.92 * 0.88
        options.hadronic_only_eff = 1.0
        options.dedicated_lim = 1 * (138./options.lumi)**0.5
        options.dedicated_label = "Dedicated Search, B2G-20-009 (%.1f fb)" % options.dedicated_lim
    elif(options.sig_idx == 3):
        options.preselection_eff = 0.76*0.59
        options.dedicated_lim = 8 * (138./options.lumi)**0.5
        options.dedicated_label = "Dedicated Search, B2G-21-002 (%.1f fb)" % options.dedicated_lim
        if(options.hadronic_only):
            options.hadronic_only_eff = 0.36
        else:
            options.hadronic_only_eff = 1.0
            
        
    else:
        print("Sig preselection stuff not computed for sig %i" % options.sig_idx)
        sys.exit(1)



def make_signif_plot(options, signifs):
    fout = options.output + options.label + "_signif_plot.png"

    options.lumi = 28.
    if(int(options.mjj_sig) in fit_results.keys()):
        n_evts_exc, n_evts_exc_no_sel = fit_results[int(options.mjj_sig)]

    else:
        print("Can't find signal mjj (%i) in fit results dict" % int(options.mjj))
        n_evts_exc = 93.
        n_evts_exc_no_sel = 1000.
    print("N_evts_exc %.0f No cut %.0f " % ( n_evts_exc, n_evts_exc_no_sel))

    get_signal_params(options)

    injected_xsecs = np.array([( spb*options.numBatches / options.lumi / options.preselection_eff / options.hadronic_only_eff) for spb in options.spbs])

    xs = injected_xsecs
    ys = signifs
    no_sel_limit = n_evts_exc_no_sel / (options.preselection_eff * options.lumi * options.hadronic_only_eff)

    no_sel_x = no_sel_limit



    fig_size = (12,9)
    plt.figure(figsize=fig_size)
    size = 0.4

    y_stop = max(ys)
    yline = np.arange(0,y_stop,y_stop/10)
    if(no_sel_x >0):
        label = "Inclusive Limit (%.1f fb)" % no_sel_x
        plt.plot([no_sel_x]*10, yline, linestyle = "--", color = "green", linewidth = 2, label = label)

    #if(options.dedicated_lim > 0):
        #plt.plot([options.dedicated_lim] * 10, yline, linestyle = "--", color = "cyan", linewidth = 2, label = options.dedicated_label)


    plt.scatter(xs, ys, c ='b' , s=40.0, label = "Injections")




    plt.ylim([0., 8.])
    plt.xlim([0., np.max(xs) * 1.2])

    plt.xlabel("Injected cross section (fb)", fontsize=20)
    plt.ylabel(r"Significance ($\sigma$)", fontsize=20)

    plt.tick_params(axis='y', labelsize=16)
    plt.tick_params(axis='x', labelsize=16)

    plt.legend(loc="upper right", fontsize = 16)

    print("saving %s" % fout)
    plt.savefig(fout)




def make_limit_plot(options, sig_effs):
    fout = options.output + options.label + "_limit_plot.png"
    if(options.num_events): fout = options.output + options.label + "_limit_plot_numevents.png"

    if(int(options.mjj_sig) in fit_results.keys()):
        n_evts_exc, n_evts_exc_no_sel = fit_results[int(options.mjj_sig)]

    else:
        print("Can't find signal mjj (%i) in fit results dict" % int(options.mjj))
        n_evts_exc = 93.
        n_evts_exc_no_sel = 1000.
    print("N_evts_exc %.0f No cut %.0f " % ( n_evts_exc, n_evts_exc_no_sel))

    get_signal_params(options)


    sig_effs = np.clip(sig_effs, 1e-4, 1.0)
    injected_xsecs = np.array([( spb*options.numBatches / options.lumi / options.preselection_eff / options.hadronic_only_eff) for spb in options.spbs])
    excluded_xsecs = np.array([(n_evts_exc / (options.hadronic_only_eff * sig_eff * options.preselection_eff * options.lumi)) for sig_eff in sig_effs])
    no_sel_limit = n_evts_exc_no_sel / (options.preselection_eff  * options.lumi * options.hadronic_only_eff)

    injected_nsig  = np.array( [spb * options.numBatches  for spb in options.spbs])
    excluded_nsig = np.array([n_evts_exc / sig_eff  for sig_eff in sig_effs])
    no_sel_limit_nsig = n_evts_exc_no_sel / (options.preselection_eff)


    if(options.num_events): 
        xs = injected_nsig
        ys = excluded_nsig
        no_sel_y = no_sel_limit_nsig
    else:
        xs = injected_xsecs
        ys = excluded_xsecs
        no_sel_y = no_sel_limit




    best_lim = 999999
    for i,x in enumerate(xs):
        y = ys[i]
        lim = max(x, y)
        if(lim < best_lim):
            best_lim = lim
            best_i = i

        print(options.spbs[i], x,y)

    if(options.num_events): 
        best_lim_label = "Best Limit (%.1f events)" % best_lim
    else:
        best_lim_label = "Best Limit (%.1f fb)" % best_lim

    print("Limit without NN selection: %.1f" % no_sel_y)
    print("Best lim : %.3f"  % best_lim)
        
    fig_size = (12,9)
    plt.figure(figsize=fig_size)
    size = 0.4

    x_stop = max(xs)
    xline = np.arange(0,x_stop,x_stop/10)
    plt.plot(xline, xline, linestyle = "--", color = "black", linewidth = 2, label = "Excluded = Injected")
    if(no_sel_y >0):
        label = "Inclusive Limit (%.1f fb)" % no_sel_y
        if(options.num_events): label = "Inclusive Limit (%.1f events)" % no_sel_y
        plt.plot(xline, [no_sel_y]*10, linestyle = "--", color = "green", linewidth = 2, label = label)
    if(options.dedicated_lim > 0):
        plt.plot(xline, [options.dedicated_lim]*10, linestyle = "--", color = "cyan", linewidth = 2, label = options.dedicated_label)




    vertical_line = xs[best_i] > ys[best_i]
    lim_line_max = min(xs[best_i], ys[best_i])*0.97

    if(vertical_line):
        plt.plot([best_lim, best_lim], [0., lim_line_max], linestyle="--", color = "red", linewidth =2, label = best_lim_label)
    else: #horizontal line
        plt.plot( [0., lim_line_max], [best_lim, best_lim], linestyle="--", color = "red", linewidth =2, label = best_lim_label)

    plt.scatter(xs, ys, c ='b' , s=40.0, label="Injections")

    off_plots = ys > (x_stop * 1.2)
    num_off = np.sum(off_plots)
    off_ys = [x_stop * 1.18] * num_off
    plt.scatter(xs[off_plots], off_ys, c = 'b', s = 40.0, marker ="^")

    plt.ylim([0., x_stop * 1.2])
    plt.xlim([0., x_stop * 1.2])

    if(not options.num_events):
        plt.xlabel("Injected cross section (fb)", fontsize=20)
        plt.ylabel(" 'Excluded' cross section (fb)", fontsize=20)
    else:
        plt.xlabel("Injected Number of Events", fontsize=20)
        plt.ylabel(" 'Excluded' Number of Events", fontsize=20)

    plt.tick_params(axis='y', labelsize=16)
    plt.tick_params(axis='x', labelsize=16)

    fs_leg = 16
    plt.legend(loc="upper right", fontsize = fs_leg)
    print("saving %s" % fout)
    plt.savefig(fout)

def make_sys_plot(options, nom_eff, sys_diffs, extra_label = ""):

    fout = options.output + options.label + extra_label + "_sys_plot.png"
    names = []
    ups = []
    downs = []

    for i in range(len(sys_diffs)):
        if(abs(sys_diffs[i][1][0]) > 0. or abs(sys_diffs[i][1][1]) > 0.):
            names.append(sys_diffs[i][0])
            ups.append(sys_diffs[i][1][0] / nom_eff)
            downs.append(sys_diffs[i][1][1] / nom_eff)

    fig_size = (25,10)
    plt.figure(figsize=fig_size)
    ax = plt.subplot(111)
    x = np.array(range(len(names))) 

    ax.bar(x, ups, width = 0.8, color = 'b', tick_label = names, alpha = 0.5, label = "Up variations")
    ax.bar(x, downs, width = 0.8, color = 'r', alpha = 0.5 , label = "Down variations")
    plt.legend(loc="upper right", fontsize = 20)

    plt.ylabel(" Fractional Change in Sig. Eff. ", fontsize=24)
    plt.xlabel(" Systematic", fontsize=24)
    plt.tick_params(axis='y', labelsize=20)
    plt.tick_params(axis='x', labelsize=20)

    print("saving %s" % fout)
    plt.savefig(fout)







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

def get_fit_results(outdir, m):
    fname = outdir + "fit_results_%.1f.pkl" % m
    if(not os.path.exists(fname)):
        print("Can't find fit inputs " + fname)
        return None

    fit_results = get_options_from_pkl(fname)
    return fit_results



def limit_set(options):

    num_rand = 4

    if(options.output[-1] != '/'):
        options.output += '/'

    if(not os.path.exists(options.output)):
        subprocess.call("mkdir %s" % options.output, shell = True)



    if(options.reload):

        if(not os.path.exists(options.output + "run_opts.pkl")):
            print("Reload options specified but file %s doesn't exist (add --new to create new directory). Exiting" % (options.output+"run_opts.pkl"))
            sys.exit(1)
        else:
            #rel_opts = get_options_from_pkl(options.output + "run_opts.pkl")
            rel_opts = get_options_from_json(options.output + "run_opts.json")
            print(rel_opts.__dict__)
            rel_opts.keep_LSF = options.keep_LSF #quick fix
            rel_opts.num_events = options.num_events #quick fix
            rel_opts.sys_train_all = options.sys_train_all #quick fix
            if(abs(options.mjj_sig - 2500.) > 1.):  rel_opts.mjj_sig  = options.mjj_sig #quick fix
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
    #write_options_to_pkl(options_dict, options.output + "run_opts.pkl" )
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
    do_roc = options.step == "roc"
    do_sys_train = options.step == "sys_train"
    do_rand_train = options.step == "rand_train"
    do_rand_select = options.step == "rand_select"
    do_rand_merge = options.step == "rand_merge"
    do_sys_merge = options.step == "sys_merge"
    do_sys_selection = options.step == "sys_select"
    do_sys_plot = options.step == "sys_plot"

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

    if(do_sys_selection): #compute eff for systematics

        if(len(spbs_to_run) == 1):
            inj_spb = spbs_to_run[0]
            print("Using spb %i" % inj_spb)
        else:
            sig_effs = []
            for spb in options.spbs:
                sig_eff = get_sig_eff(options.output + "spb" + str(float(spb)) + "/", eff = options.effs[0])
                sig_effs.append(sig_eff)

            inj_spb = get_optimal_spb(options, sig_effs)

        #add JME vars but remove duplicates
        sys_select_list = list(set(options.saved_params['sys_train_list'] + list(JME_vars)))

        options.saved_params['sys_select_list'] = sys_select_list



        t_opts_orig = spb_opts(options, inj_spb)
        for sys in sys_select_list:
            t_opts = spb_opts(options, inj_spb, sys = sys)
            if(sys in options.saved_params['sys_train_list']):
            #if(sys not in JME_vars):
                t_opts.reload = True
            else:
                t_opts.reload = False
                t_opts.new = True
                #for now copy everything, TODO make more memory efficienct using sym links for models
                os.system("cp -r %s %s" % (t_opts_orig.output, t_opts.output))
                
            t_opts.step = "select"
            t_opts.eff_only = True
            t_opts.condor = True
            full_run(t_opts)

        for seed in range(num_rand):
            t_opts = spb_opts(options, inj_spb, sys = "rand%i" % seed)
            t_opts.step = "select"
            t_opts.eff_only = True
            t_opts.reload = True
            #t_opts.new = True
            t_opts.condor = True
            full_run(t_opts)

    if(do_sys_plot):

        sig_effs = []
        for spb in options.spbs:
            sig_eff = get_sig_eff(options.output + "spb" + str(float(spb)) + "/", eff = options.effs[0])
            sig_effs.append(sig_eff)

        if(len(spbs_to_run) == 1):
            inj_spb = spbs_to_run[0]
            print("Using spb %i" % inj_spb)
        else:

            inj_spb = get_optimal_spb(options, sig_effs)



        sig_eff_nom = sig_eff_nom_fixed = sig_effs[options.spbs.index(inj_spb)]
        #dictionary of change in eff
        diffs_dict_retrain = dict() #when retrained
        diffs_dict_fixed = dict() #for fixed training
        diffs_dict_final = dict() #final combo
        #initialize
        for sys_clean in sys_list_clean:
            diffs_dict_retrain[sys_clean] = [0., 0.] 
            diffs_dict_fixed[sys_clean] = [0., 0.] 

        if(os.path.exists(options.output + "rand0/")):
            rand_effs = [sig_eff_nom]
            #rand_effs = []
            for seed in range(num_rand):
                eff = get_sig_eff(options.output + "rand%i" % seed  + "/", eff = options.effs[0])
                rand_effs.append(eff)

            #sig_eff_nom = np.median(rand_effs)
            sig_eff_nom = np.mean(rand_effs)
            rand_down =  np.min(rand_effs) - sig_eff_nom
            rand_up = np.max(rand_effs) - sig_eff_nom
            diffs_dict_retrain['Rand. Var.'] = [rand_up, rand_down]

        for sys in sys_list:
            sys_retrained = sys in options.saved_params['sys_train_list']
            sys_eff_retrain = 0.
            sys_eff_fixed = 0.
            if(sys in JME_vars):
                eff = get_sig_eff(options.output + sys + "/", eff = options.effs[0])
                if(sys_retrained): 
                    sys_eff_retrain = eff
                    sys_eff_fixed = get_sig_eff(options.output + sys + "_fixed/", eff = options.effs[0])
                else: sys_eff_fixed = eff

            else:
                if(sys_retrained): sys_eff_retrain = get_sig_eff(options.output + sys + "/", eff = options.effs[0])
                sys_eff_fixed = get_sig_eff(options.output + "spb" + str(float(inj_spb)) + "/", eff = options.effs[0], sys = sys)


            if(sys_eff_retrain > 0.): diff_retrain =  sys_eff_retrain - sig_eff_nom
            else: diff_retrain = 0.

            #Fixed training, use nominal efficiency from that particular random seed (as opposed to average across rand. variations)
            if(sys_eff_fixed > 0.): diff_fixed =  sys_eff_fixed - sig_eff_nom_fixed
            else: diff_fixed = 0.


            sys_clean = sys.replace("_up", "").replace("_down", "")
            if('_up' in sys) : 
                diffs_dict_fixed[sys_clean][0] = diff_fixed
                diffs_dict_retrain[sys_clean][0] = diff_retrain
            elif('_down' in sys) : 
                diffs_dict_fixed[sys_clean][1] = diff_fixed
                diffs_dict_retrain[sys_clean][1] = diff_retrain
            else: print("Sys is %s ? " % sys)

            
        print("Nominal eff is %.4f " % sig_eff_nom)
        #sort by avg difference (largest first)
        sort_fn = lambda kv: -(abs(kv[1][0]) + abs(kv[1][1]))/ 2.0
        diffs_retrain_sorted = sorted(diffs_dict_retrain.items(), key = sort_fn) 
        diffs_fixed_sorted = sorted(diffs_dict_fixed.items(), key = sort_fn) 


        print("Fixed", diffs_fixed_sorted, "\n")
        print("Retrain", diffs_retrain_sorted, "\n")

        make_sys_plot(options, sig_eff_nom, diffs_fixed_sorted, extra_label = "_fixed")
        make_sys_plot(options, sig_eff_nom, diffs_retrain_sorted, extra_label = "_retrain")
        

        for key in diffs_dict_retrain.keys():
            if('Rand' in key):
                diffs_dict_final[key] = diffs_dict_retrain[key]
            
            elif(key + "_up" in options.saved_params['sys_train_list']):
                #take larger variation
                if(abs(diffs_dict_retrain[key][0]) > abs(diffs_dict_fixed[key][0])): up = diffs_dict_retrain[key][0]
                else: up = diffs_dict_fixed[key][0]
                if(abs(diffs_dict_retrain[key][1]) > abs(diffs_dict_fixed[key][1])): down = diffs_dict_retrain[key][1]
                else: down = diffs_dict_fixed[key][1]

                diffs_dict_final[key] = [up, down]
            else:
                diffs_dict_final[key] = diffs_dict_fixed[key]

        diffs_final_sorted = sorted(diffs_dict_final.items(), key = sort_fn) 
        make_sys_plot(options, sig_eff_nom, diffs_final_sorted, extra_label = "_final")

        up_tot = down_tot = 0.
        for entry in diffs_final_sorted:
            up_tot += entry[1][0] **2
            down_tot += entry[1][1] **2

        up_tot = up_tot ** 0.5
        down_tot = down_tot ** 0.5

        print(" \n Eff final %.3f + %.3f - %.3f \n" % (sig_eff_nom, up_tot, down_tot))


        frac_unc = (up_tot + down_tot) / (2. * sig_eff_nom)
        #frac_unc = 0.0001


        #for now use expected lim from fit to injected, eventually switch to fit to data/signaless mc
        t_opts = spb_opts(options, inj_spb)
        t_opts.step = "fit"
        t_opts.sig_norm_unc = frac_unc
        t_opts.reload = False
        t_opts.condor = False
        full_run(t_opts)
        fit_results = get_fit_results(t_opts.output, options.mjj_sig)
        #print(fit_results)
        n_evts_exc = fit_results.exp_lim_events
        get_signal_params(options)

        injected_xsec = inj_spb*options.numBatches / options.lumi / options.preselection_eff / options.hadronic_only_eff
        excluded_xsec = n_evts_exc / (options.hadronic_only_eff * sig_eff_nom * options.preselection_eff * options.lumi)

        print("Final result: Injected %.2f Excluded %.2f" % (injected_xsec, excluded_xsec))


    if(do_merge):
        for spb in options.spbs:
            t_opts = spb_opts(options, spb)
            t_opts.step = "merge"
            t_opts.reload = True
            t_opts.condor = True
            full_run(t_opts)


    if(do_sys_merge):
        if(len(spbs_to_run) == 1):
            inj_spb = spbs_to_run[0]
            print("Using spb %i" % inj_spb)
        else:
            sig_effs = []
            for spb in options.spbs:
                sig_eff = get_sig_eff(options.output + "spb" + str(float(spb)) + "/", eff = options.effs[0])
                sig_effs.append(sig_eff)

            inj_spb = get_optimal_spb(options, sig_effs)

        sys_select_list = options.saved_params["sys_select_list"]

        for sys in sys_select_list:
            t_opts = spb_opts(options, inj_spb, sys = sys)
            t_opts.step = "merge"
            t_opts.eff_only = True
            t_opts.reload = True
            #t_opts.new = True
            t_opts.condor = True
            full_run(t_opts)

        for seed in range(num_rand):
            t_opts = spb_opts(options, inj_spb, sys = "rand%i" % seed)
            t_opts.step = "merge"
            t_opts.eff_only = True
            t_opts.reload = True
            #t_opts.new = True
            t_opts.condor = True
            full_run(t_opts)

    if(do_fit):
        for spb in options.spbs:
            t_opts = spb_opts(options, spb)
            t_opts.step = "fit"
            t_opts.reload = True
            t_opts.condor = False
            full_run(t_opts)

    if(do_roc):
        for spb in options.spbs:
            t_opts = spb_opts(options, spb)
            t_opts.step = "roc"
            t_opts.reload = True
            t_opts.condor = False
            full_run(t_opts)

    if(do_plot):
        sig_effs = []
        signifs = []


        for spb in options.spbs:
            sig_eff = float(get_sig_eff(options.output + "spb" + str(float(spb)) + "/", eff = options.effs[0]))
            fit_res = get_fit_results(options.output + "spb" + str(float(spb)) + "/",  options.mjj_sig)
            if(fit_res is not None): signif = float(fit_res.signif)
            else: signif = 0.

            sig_effs.append(sig_eff)
            signifs.append(signif)

        print(sig_effs)
        print(signifs)

        options.saved_params['signifs'] = signifs
        options.saved_params['sig_effs'] = sig_effs

        sig_effs = np.array(sig_effs)
        np.savez(f_sig_effs, sig_eff = sig_effs)

        make_limit_plot(options, sig_effs)

        if(signifs[-1] > 0):
            make_signif_plot(options, signifs)
    if(do_sys_train):
        if(len(spbs_to_run) == 1):
            inj_spb = spbs_to_run[0]
            print("Using spb %i" % inj_spb)
        else:
            sig_effs = []
            for spb in options.spbs:
                sig_eff = get_sig_eff(options.output + "spb" + str(float(spb)) + "/", eff = options.effs[0])
                sig_effs.append(sig_eff)

            inj_spb = get_optimal_spb(options, sig_effs)


        if(options.sys_train_all): sys_train_list = sys_list
        else: 
            t_opts = spb_opts(options, inj_spb)
            t_opts.output += "sig_sys_plots/"
            sys_train_var_list = draw_sys_variations(t_opts)
            sys_train_list = []
            for s in sys_train_var_list:
                sys_train_list.append(s+"_up")
                sys_train_list.append(s+"_down")



        print(sys_train_list)

        options.saved_params['sys_train_list'] = list(sys_train_list)


        for sys in sys_train_list:
            t_opts = spb_opts(options, inj_spb, sys = sys)
            t_opts.step = "train"
            t_opts.condor = True
            #full_run(t_opts)


        #do rand trainings too
        for seed in range(num_rand):
            t_opts = spb_opts(options, inj_spb, sys = "rand%i" % seed)
            t_opts.BB_seed = seed
            t_opts.step = "train"
            t_opts.condor = True
            #full_run(t_opts)

    write_params(options.output + "saved_params.json", options.saved_params)




if(__name__ == "__main__"):
    parser = input_options()
    parser.add_argument("--sig_norm_unc", default = -1.0, type = float, help = "parameter for fit (uncertainty on signal efficiency)")
    parser.add_argument("--spbs", nargs="+", default = [], type = float)
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
    options = parser.parse_args()
    limit_set(options)
