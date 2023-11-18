from full_run import *
from plotting.draw_sys_variations import *
import sys

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
    t_opts.saved_params = None
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

def get_optimal_spb(options):


    if('handpicked_spb' in options.saved_params.keys()):
        return options.saved_params['handpicked_spb']

    elif('best_spb' in options.saved_params.keys()):
        return options.saved_params['best_spb']

    sig_effs = []

    for spb in options.spbs:
        sig_eff = get_sig_eff(options.output + "spb" + str(float(spb)) + "/", eff = options.effs[0])
        sig_effs.append(sig_eff)

    observed = True
    if(observed):
        n_evts_exc = options.saved_params['n_evts_exc_obs']
        n_evts_exc_nosel = options.saved_params['n_evts_exc_nosel']
    else:
        n_evts_exc = options.saved_params['n_evts_exc_exp']
        n_evts_exc_nosel = options.saved_params['n_evts_exc_nosel_exp']


    print("N_evts_exc %.0f No cut %.0f " % ( n_evts_exc, n_evts_exc_nosel))

    get_signal_params(options)

    sig_effs = np.clip(sig_effs, 1e-4, 1.0)
    injected_xsecs = np.array([( spb*options.numBatches / options.lumi / options.preselection_eff / options.hadronic_only_eff) for spb in options.spbs])
    excluded_xsecs = np.array([(n_evts_exc / (options.hadronic_only_eff * sig_eff * options.preselection_eff * options.lumi)) for sig_eff in sig_effs])

    lims = [ max(injected_xsecs[i], excluded_xsecs[i]) for i in range(len(options.spbs))]
    print("xsec lims are : ", lims)
    optimal_spb = options.spbs[np.argmin(lims)]
    print("Optimal spb %i" % optimal_spb)
    return optimal_spb

def get_preselection_params(sig_fname, hadronic_only_cut = False, sig_mass = 2500.):
    print("Getting preselection params from %s" % sig_fname)
    if(not os.path.exists(sig_fname)):
        sig_fname = sig_fname.replace("data", "data/LundRW")
    if(not os.path.exists(sig_fname)):
        print("Can't find file " + sig_fname)
        exit(1)

    f = h5py.File(sig_fname, "r")
    is_lep = f['event_info'][:,4]
    weights = f['sys_weights'][:,0]
    mjj = f['jet_kinematics'][:,0]
    deta = f['jet_kinematics'][:,1]

    presel_eff = f['preselection_eff'][0]
    deta_eff = f['d_eta_eff'][0]
    deta_mask = np.abs(deta) < 1.3
    mjj_mask = (mjj  > 0.8 * sig_mass)  & (mjj < 1.2 * sig_mass)
    mjj_lowcut = mjj > 1460.

    mjj_lowcut_eff = np.sum(weights[deta_mask & mjj_lowcut])/ np.sum(weights[deta_mask])

    print(presel_eff, deta_eff, mjj_lowcut_eff)

    hadronic_only = 1.0 - np.mean(is_lep)
    hadronic_only_mask = is_lep < 0.1
    if(hadronic_only_cut): mjj_window_eff = float(np.sum(weights[mjj_mask & hadronic_only_mask]) / np.sum(weights[hadronic_only_mask]))
    else: mjj_window_eff = float(np.sum(weights[mjj_mask]) / np.sum(weights))
    f.close()
    return float(presel_eff * deta_eff * mjj_lowcut_eff), float(hadronic_only), float(mjj_window_eff)

def get_fit_nosel_params(options):
    base_path = os.path.abspath(".") + "/"
    if(options.data): nosel_fname = base_path + "../data/fit_inputs_DATA_nosel.h5"
    else: nosel_fname = base_path + "../data/fit_inputs_nosel.h5"
    plot_dir = base_path + options.output + 'sig_nosel_fit/'
    fit_file = plot_dir + 'fit_results_%.1f.json' % options.mjj_sig
    if(not os.path.exists(plot_dir)): os.system("mkdir %s" % plot_dir)
    if(len(options.effs) == 0): options.effs = [mass_bin_select_effs[options.mbin] ]

    if(not os.path.exists(options.sig_file)): sig_file = options.sig_file.replace("data", "data/LundRW")
    else: sig_file = options.sig_file

    if(options.refit or not os.path.exists(fit_file)):
        print("Running fit with no selection \n")
        f_sig = h5py.File(sig_file, "r")
        mjj = f_sig['jet_kinematics'][:,0]
        deta = f_sig['jet_kinematics'][:,1]
        weights = f_sig['sys_weights'][:,0]
        mask = deta < options.deta
        if(options.hadronic_only):
            is_lep = f_sig['event_info'][:,4]
            hadronic_only_mask = is_lep < 0.1
            mask = mask & hadronic_only_mask

        sig_nosel_fname = options.output + "sig_shape_nosel.h5"
        with h5py.File(sig_nosel_fname, "w") as f_sig_shape:
            f_sig_shape.create_dataset('mjj', data = mjj[mask], chunks = True, maxshape = (None))
            f_sig_shape.create_dataset('weights', data = weights[mask], chunks = True, maxshape = (None))
            f_sig_shape.create_dataset('truth_label', data = np.ones((mjj[mask].shape[0],1)), chunks = True, maxshape = (None, 1))



        sig_fit_cmd = "python fit_signalshapes.py -i %s -o %s -M %i --dcb-model --fitRange 1.0 >& %s/sig_fit_log.txt" % (base_path + sig_nosel_fname, plot_dir , options.mjj_sig, plot_dir)
        print(sig_fit_cmd)
        fit_cmd_setup = "cd ../fitting; source deactivate; eval `scramv1 runtime -sh`;"  
        fit_cmd_after = "cd -; source deactivate; source activate mlenv0"
        full_sig_fit_cmd = fit_cmd_setup + sig_fit_cmd + "; "  + fit_cmd_after

        subprocess.call(full_sig_fit_cmd,  shell = True, executable = '/bin/bash')
        sig_shape_file = plot_dir + 'sig_fit_%i.root' % options.mjj_sig

        run_dijetfit(options, fit_start = -1, input_file = nosel_fname, output_dir = plot_dir, sig_shape_file = sig_shape_file, sig_norm = options.sig_norm, loop = True)

    with open(fit_file, 'r') as f:
        fit_params = json.load(f, encoding="latin-1")
        n_evts_exc_nosel = fit_params['obs_lim_events'] 

    print("No selection num events lim %.0f "  % n_evts_exc_nosel)
    return fit_params



    

def get_signal_params(options):
    #modifies options struct
    options.lumi = 26.81 if not options.data else 138.0
    options.dedicated_lim = -1
    options.dedicated_label = ""
    options.mjj_window_nosel_eff = 1.0
    if(len(options.sig_file) > 0):

        options.preselection_eff, hadronic_only_, options.mjj_window_nosel_eff = get_preselection_params(options.sig_file, options.hadronic_only, options.mjj_sig)
        
        fit_nosel = get_fit_nosel_params(options)

        options.saved_params['n_evts_exc_nosel'] =  fit_nosel['obs_lim_events']
        options.saved_params['n_evts_exc_nosel_exp'] =  fit_nosel['exp_lim_events']
        options.saved_params['n_evts_exc_nosel_exp_1sig_high'] =  fit_nosel['exp_lim_1sig_high']
        options.saved_params['n_evts_exc_nosel_exp_1sig_low'] =  fit_nosel['exp_lim_1sig_low']
        options.saved_params['fit_nosel'] =  fit_nosel

        
        if(options.hadronic_only): options.hadronic_only_eff = hadronic_only_
        else: options.hadronic_only_eff = 1.0
        sig_fn = options.sig_file.split('/')[-1]
            
        
    else:
        print("Sig preselection stuff not computed for sig %i" % options.sig_idx)
        sys.exit(1)



def make_signif_plot(options, signifs, spbs):
    fout = options.output + "plots/" + options.label + "_signif_plot.png"

    options.lumi = 26.81 if not options.data else 138.0
    #n_evts_exc, n_evts_exc_nosel = fit_results[int(options.mjj_sig)]
    n_evts_exc = options.saved_params['n_evts_exc_obs']
    n_evts_exc_nosel = options.saved_params['n_evts_exc_nosel']


     


    print("N_evts_exc %.0f No cut %.0f " % ( n_evts_exc, n_evts_exc_nosel))


    injected_xsecs = [( spb*options.numBatches / options.lumi / options.preselection_eff / options.hadronic_only_eff) for spb in spbs]
    options.saved_params['injected_xsecs'] = injected_xsecs

    xs = np.array(injected_xsecs)
    ys = signifs
    #print('window_eff', options.mjj_window_nosel_eff)
    #nosel_limit = n_evts_exc_nosel / (options.preselection_eff * options.mjj_window_nosel_eff * options.lumi * options.hadronic_only_eff)
    nosel_limit = n_evts_exc_nosel / (options.preselection_eff *  options.lumi * options.hadronic_only_eff)
    print(nosel_limit, n_evts_exc_nosel, options.preselection_eff, options.lumi, options.hadronic_only_eff)




    fig_size = (12,9)
    plt.figure(figsize=fig_size)
    size = 0.4
    linewidth = 4
    pointsize = 70.

    x_stop = max(np.max(xs), nosel_limit)
    y_stop = max(np.max(ys), 6.)

    #print('stops', x_stop, y_stop)
    #print(xs, nosel_limit) 
    #print(ys)

    yline = np.arange(0,y_stop,y_stop/10)
    if(nosel_limit >0):
        label = "Inclusive Limit (%.1f fb)" % nosel_limit
        plt.plot([nosel_limit]*10, yline, linestyle = "--", color = "green", linewidth = linewidth, label = label)

    #if(options.dedicated_lim > 0):
        #plt.plot([options.dedicated_lim] * 10, yline, linestyle = "--", color = "cyan", linewidth = 2, label = options.dedicated_label)


    plt.scatter(xs, ys, c ='b' , s=pointsize, label = "Injections")



            
    plt.ylim([0., y_stop * 1.2])
    plt.xlim([0., x_stop * 1.2])

    plt.xlabel("Injected cross section (fb)", fontsize=30)
    plt.ylabel(r"Significance ($\sigma$)", fontsize=30)

    plt.tick_params(axis='y', labelsize=18)
    plt.tick_params(axis='x', labelsize=18)

    plt.legend(loc="upper left", fontsize = 20)

    print("saving %s" % fout)
    plt.savefig(fout)


def convert_to_xsec(options, nevts, sig_eff):
    return nevts / (options.hadronic_only_eff * sig_eff * options.preselection_eff * options.lumi)


def get_excluded_xsecs(options, nevts_exc, sig_effs):
    return np.array([ convert_to_xsec(options, nevts_exc, eff) for eff in sig_effs])

def output_json(options):
    sig_name = options.sig_file.split("/")[-1]

    outfile = options.output + sig_name.replace("_Lund.h5", ".json")
    #if("XToYY" in outfile): outfile += "_narrow"
    #outfile += "_TuneCP5_13TeV-madgraph-pythia8_TIMBER.json"
    results = dict()

    results['obs'] = options.saved_params['xsec_exc_obs'] 
    results['exp'] = options.saved_params['xsec_exc_exp']
    results['exp+1'] = options.saved_params['xsec_exc_exp_1sig_high'] 
    results['exp-1'] = options.saved_params['xsec_exc_exp_1sig_low'] 
    print("Writing limits to %s" % outfile)
    write_params(outfile, results)

    if('xsec_exc_obs_sys' in options.saved_params.keys()):
        outfile_sys = outfile + '.sys'

        results_sys = dict()
        results_sys['obs'] = options.saved_params['xsec_exc_obs_sys'] 
        results_sys['exp'] = options.saved_params['xsec_exc_exp_sys']
        results_sys['exp+1'] = options.saved_params['xsec_exc_exp_1sig_high_sys'] 
        results_sys['exp-1'] = options.saved_params['xsec_exc_exp_1sig_low_sys'] 
        print("Writing limits with systematics to %s" % outfile_sys)
        write_params(outfile_sys, results_sys)


    if('inc_xsec_exc_obs' in options.saved_params.keys()):
        outfile_inc = outfile + '.inc'

        results_sys = dict()
        results_sys['obs'] = options.saved_params['inc_xsec_exc_obs'] 
        results_sys['exp'] = options.saved_params['inc_xsec_exc_exp']
        results_sys['exp+1'] = options.saved_params['inc_xsec_exc_exp_1sig_high'] 
        results_sys['exp-1'] = options.saved_params['inc_xsec_exc_exp_1sig_low'] 
        print("Writing inclusive limits to %s" % outfile_inc)
        write_params(outfile_inc, results_sys)

    print("\n")

def make_sig_eff_plot(options, sig_effs, spbs):

    injected_xsecs = np.array([( spb*options.numBatches / options.lumi / options.preselection_eff / options.hadronic_only_eff) for spb in spbs])

    pointsize = 100
    fig_size = (12,9)
    fout = options.output + "plots/" + options.label + "_sig_eff_plot.png"
    plt.figure(figsize=fig_size)

    plt.scatter( injected_xsecs, sig_effs, color = "blue", s = pointsize)
    plt.ylim([0, None])
    plt.xlim([-0.2, None])

    plt.tick_params(axis='y', labelsize=18)
    plt.tick_params(axis='x', labelsize=18)
    plt.xlabel("Cross section (fb)", fontsize=30)
    plt.ylabel("Signal Efficiency", fontsize=30)


    #text_str = 'TNT : X -> YY, 3 TeV'
    text_str = "TNT : W' -> B't, 3 TeV"

    plt.annotate(text_str, xy = (0.03, 0.9),  xycoords = 'axes fraction', fontsize=24)

    plt.savefig(fout, bbox_inches = 'tight')




def make_limit_plot(options, sig_effs, spbs):
    fout = options.output + "plots/" + options.label + "_limit_plot.png"
    if(options.num_events): fout = options.output + "plots/" + options.label + "_limit_plot_numevents.png"

    n_evts_exc_obs = options.saved_params['n_evts_exc_obs']
    n_evts_exc_exp = options.saved_params['n_evts_exc_exp']
    n_evts_exc_nosel = options.saved_params['n_evts_exc_nosel']
    n_evts_exc_nosel_exp= options.saved_params['n_evts_exc_nosel_exp']
    n_evts_exc_nosel_exp_up= options.saved_params['n_evts_exc_nosel_exp_1sig_high']
    n_evts_exc_nosel_exp_down= options.saved_params['n_evts_exc_nosel_exp_1sig_low']

    print("N_evts_exc %.0f No cut %.0f " % ( n_evts_exc_obs, n_evts_exc_nosel))

    get_signal_params(options)

    print("presel eff",  options.preselection_eff)

    sig_effs = np.clip(sig_effs, 1e-4, 1.0)
    injected_xsecs = np.array([( spb*options.numBatches / options.lumi / options.preselection_eff / options.hadronic_only_eff) for spb in spbs])
    nosel_limit = n_evts_exc_nosel / (options.preselection_eff  * options.lumi * options.hadronic_only_eff)
    nosel_limit_exp = n_evts_exc_nosel_exp / (options.preselection_eff  * options.lumi * options.hadronic_only_eff)
    nosel_limit_exp_up = n_evts_exc_nosel_exp_up / (options.preselection_eff  * options.lumi * options.hadronic_only_eff)
    nosel_limit_exp_down = n_evts_exc_nosel_exp_down / (options.preselection_eff  * options.lumi * options.hadronic_only_eff)

    obs_excluded_xsecs = get_excluded_xsecs(options, options.saved_params['n_evts_exc_obs'], sig_effs)
    exp_excluded_xsecs = get_excluded_xsecs(options, options.saved_params['n_evts_exc_exp'], sig_effs)
    exp_excluded_xsecs_1sig_high = get_excluded_xsecs(options, options.saved_params['n_evts_exc_exp_1sig_high'], sig_effs)
    exp_excluded_xsecs_1sig_low = get_excluded_xsecs(options, options.saved_params['n_evts_exc_exp_1sig_low'], sig_effs)



    best_lim_obs = 999999
    best_lim_exp = 999999
    for i,x_inj in enumerate(injected_xsecs):
        x_exc_obs = obs_excluded_xsecs[i]
        x_exc_exp = exp_excluded_xsecs[i]
        obs_lim = max(x_inj, x_exc_obs)
        exp_lim = max(x_inj, x_exc_exp)

        if(obs_lim < best_lim_obs):
            best_lim_obs = obs_lim
            best_i_obs = i
        if(exp_lim < best_lim_exp):
            best_lim_exp = exp_lim
            best_i_exp = i

        print(spbs[i], x_inj, x_exc_obs, x_exc_exp)


    best_i = best_i_exp
    #best_i = best_i_obs

    options.saved_params['best_sig_eff'] = sig_effs[best_i]
    options.saved_params['best_lim'] = best_lim_obs
    options.saved_params['best_spb'] = spbs[best_i]
    options.saved_params['best_obs_spb'] = spbs[best_i_obs]
    options.saved_params['best_exp_spb'] = spbs[best_i_exp]

    options.saved_params['inc_xsec_exc_obs'] = nosel_limit
    options.saved_params['inc_xsec_exc_exp'] = nosel_limit_exp
    options.saved_params['inc_xsec_exc_exp_up'] = nosel_limit_exp
    options.saved_params['inc_xsec_exc_exp_1sig_high'] = nosel_limit_exp_up
    options.saved_params['inc_xsec_exc_exp_1sig_low'] = nosel_limit_exp_down

    options.saved_params['xsec_exc_obs'] = max(obs_excluded_xsecs[best_i_obs], injected_xsecs[best_i_obs])
    options.saved_params['xsec_exc_exp'] = max(exp_excluded_xsecs[best_i_exp], injected_xsecs[best_i_exp])
    options.saved_params['xsec_exc_exp_1sig_high'] = max(exp_excluded_xsecs_1sig_high[best_i_exp], injected_xsecs[best_i_exp])
    options.saved_params['xsec_exc_exp_1sig_low'] = max(exp_excluded_xsecs_1sig_low[best_i_exp], injected_xsecs[best_i_exp])
        
    fig_size = (12,9)
    plt.figure(figsize=fig_size)
    size = 0.4
    linewidth = 4
    pointsize = 70.

    xs = injected_xsecs
    ys = obs_excluded_xsecs
    nosel_y = nosel_limit

    best_lim_label = "Best Limit %.1f (obs)" % best_lim_obs
    print("Limit without NN selection: %.1f (obs) %.1f (exp)" % (nosel_limit, nosel_limit_exp))
    print("Best Limit %.2f (obs) %.2f (exp)" % (best_lim_obs, best_lim_exp))

    x_stop = max(np.max(xs), nosel_limit)
    xline = np.arange(0,x_stop,x_stop/10)
    plt.plot(xline, xline, linestyle = "--", color = "black", linewidth = linewidth, label = "Excluded = Injected")
    if(nosel_y >0):
        label = "Inclusive Limit (%.1f fb)" % nosel_y
        if(options.num_events): label = "Inclusive Limit (%.1f events)" % nosel_y
        plt.plot(xline, [nosel_y]*10, linestyle = "--", color = "green", linewidth = linewidth, label = label)
    if(options.dedicated_lim > 0):
        plt.plot(xline, [options.dedicated_lim]*10, linestyle = "--", color = "cyan", linewidth = linewidth, label = options.dedicated_label)




    vertical_line = xs[best_i] > ys[best_i]
    lim_line_max = min(xs[best_i], ys[best_i])*0.97

    if(vertical_line):
        plt.plot([best_lim_obs, best_lim_obs], [0., lim_line_max], linestyle="--", color = "red", linewidth =linewidth, label = best_lim_label)
    else: #horizontal line
        plt.plot( [0., lim_line_max], [best_lim_obs, best_lim_obs], linestyle="--", color = "red", linewidth =linewidth, label = best_lim_label)

    plt.scatter(xs, ys, c ='b' , s=pointsize, label="Injections")

    off_plots = ys > (x_stop * 1.2)
    num_off = np.sum(off_plots)
    off_ys = [x_stop * 1.18] * num_off
    plt.scatter(xs[off_plots], off_ys, c = 'b', s = pointsize, marker ="^")

    plt.ylim([0., x_stop * 1.2])
    plt.xlim([0., x_stop * 1.2])

    if(not options.num_events):
        plt.xlabel("Injected cross section (fb)", fontsize=30)
        plt.ylabel(" 'Excluded' cross section (fb)", fontsize=30)
    else:
        plt.xlabel("Injected Number of Events", fontsize=30)
        plt.ylabel(" 'Excluded' Number of Events", fontsize=30)

    plt.tick_params(axis='y', labelsize=18)
    plt.tick_params(axis='x', labelsize=18)

    fs_leg = 20
    plt.legend(loc="upper right", fontsize = fs_leg, framealpha = 1.0)
    print("saving %s" % fout)
    plt.savefig(fout)

def make_sys_plot(options, nom_eff, sys_diffs, extra_label = ""):

    fout = options.output + "plots/" + options.label + extra_label + "_sys_plot.png"
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

    ax.bar(x, ups, width = 0.8, color = 'b', alpha = 0.5, label = "Up variations")
    ax.bar(x, downs, width = 0.8, color = 'r', alpha = 0.5 , label = "Down variations")
    plt.xticks(x, names, rotation = 'vertical')
    plt.legend(loc="upper right", fontsize = 20)

    plt.ylabel(" Fractional Change in Sig. Eff. ", fontsize=30)
    plt.xlabel(" Systematic", fontsize=24)
    plt.tick_params(axis='y', labelsize=20)
    plt.tick_params(axis='x', labelsize=16)

    print("saving %s" % fout)
    plt.savefig(fout, bbox_inches= 'tight')





def get_matching_unc(sig_file):
    if(not os.path.exists(sig_file)):
        sig_file = sig_file.replace("data/", "data/LundRW/")
    with h5py.File(sig_file, "r") as f:
        x = f['lund_weights_matching_unc'][0]
    return x

def get_sig_eff(outdir, eff = 1.0, sys = "", noprint = False):
    fname = outdir + "fit_inputs_eff{eff}.h5".format(eff = eff)
    #eff_key = 'sig_eff_window' #with mjj window eff (for counting based limit)
    eff_key = 'sig_eff' #no mjj window (for shape based limit)
    if(len(sys) > 0): eff_key += "_" + sys
    if(not os.path.exists(fname)):
        if(not noprint): print("Can't find fit inputs " + fname)
        return 0.

    with h5py.File(fname, "r") as f:

        if(eff_key not in f.keys()): 
            print("Missing key %s" % eff_key)
            return 0.

        sig_eff = f[eff_key][0]
        #print(eff_key, sig_eff)

        return sig_eff

def get_fit_data_file(options):
    fit_start, fit_stop = -1, -1
    if(options.do_TNT):
        f_base = "/uscms_data/d3/oamram/CASE_analysis/src/CASE/TagNTrain/runs/TNT_DATA_scan_june9/"
    else:
        f_base = "/uscms_data/d3/oamram/CASE_analysis/src/CASE/TagNTrain/runs/cwola_DATA_scan_june9/"
        if(options.mbin == 13):
            fit_start, fit_stop = 1650, 6000
    f_base += "mbin%i/fit_inputs_eff%.1f.h5" % (options.mbin, options.effs[0])
    return f_base, fit_start, fit_stop


def get_data_fit_results(options, sig_norm_unc = -1.0, update = False):
    get_fit_nosel_params(options)
    fname = options.output + "data_fit_results.json" if sig_norm_unc < 0. else options.output + "data_fit_results_sys.json"

    if(os.path.exists(fname) and (sig_norm_unc < 0 or abs(get_options_from_json(fname).script_options['sig_norm_unc'] - sig_norm_unc) < 0.01)):
        #if fit file already exists, with same signal normalization unc, don't rerun
        fit_res = get_options_from_json(fname)
    else:
        # fit this signal template to the selected data
        if(sig_norm_unc > 1.0):
            print("SIG NORM UNCERTAINTY %.2f! Exiting \n\n")
            exit(1)
        options.sig_norm_unc = sig_norm_unc
        base_path = os.path.abspath(".") + "/"
        sig_shape_file = base_path + options.output + "sig_nosel_fit/sig_fit_%i.root" % options.mjj_sig
        data_file, fit_start, fit_stop = get_fit_data_file(options)
        plot_dir = base_path + options.output + ("sig_selected_fit/" if sig_norm_unc < 0. else "sig_selected_sys_fit/")
        os.system("mkdir " + plot_dir)

        run_dijetfit(options, fit_start = fit_start, fit_stop = fit_stop, input_file = data_file, output_dir = plot_dir, sig_shape_file = sig_shape_file, 
                sig_norm = options.sig_norm, loop = True)
        print_and_do("cp %s %s" % (plot_dir + "fit_results_%.1f.json" % options.mjj_sig, fname))

        fit_res = get_options_from_json(fname)
    return fit_res


def get_fit_results(options = None, outdir = "", m = 3000.):
    if(len(outdir) > 0): fname = outdir + "fit_results_%.1f.json" % m
    else:
        TNT_base = "/uscms_data/d3/oamram/CASE_analysis/src/CASE/TagNTrain/runs/TNT_scan_nov19/"
        cwola_base = "/uscms_data/d3/oamram/CASE_analysis/src/CASE/TagNTrain/runs/cwola_scan_nov19/"
        if(options.do_TNT):
            fname = TNT_base + ("mbin%i/" % options.mbin) + "fit_results_%.1f.json" % m
        else:
            fname = cwola_base + ("mbin%i/" % options.mbin) + "fit_results_%.1f.json" % m

    if(not os.path.exists(fname)):
        print("Can't find fit inputs " + fname)
        return None

    fit_results = get_options_from_json(fname)
    return fit_results



def limit_set(options):
    if(len(options.label) == 0):
        if(options.output[-1] == "/"):
            options.label = options.output.split("/")[-2]
        else:
            options.label = options.output.split("/")[-1]

    num_rand = 4

    if(options.output[-1] != '/'):
        options.output += '/'

    if(not os.path.exists(options.output)):
        subprocess.call("mkdir %s" % options.output, shell = True)



    if(options.reload):

        if(not os.path.exists(options.output + "run_opts.json")):
            print("Reload options specified but file %s doesn't exist (add --new to create new directory). Exiting" % (options.output+"run_opts.json"))
            exit(1)
        else:
            rel_opts = get_options_from_json(options.output + "run_opts.json")
            #print(rel_opts.__dict__)
            rel_opts.keep_LSF = options.keep_LSF #quick fix
            rel_opts.num_events = options.num_events #quick fix
            rel_opts.sys_train_all = options.sys_train_all #quick fix
            if(abs(options.mjj_sig - 2500.) > 1.):  rel_opts.mjj_sig  = options.mjj_sig #quick fix
            rel_opts.condor_mem = options.condor_mem
            rel_opts.step = options.step
            rel_opts.condor = options.condor
            rel_opts.retrain = options.retrain
            rel_opts.seed = options.seed
            rel_opts.lund_weights = True
            rel_opts.BB_seed = options.BB_seed
            rel_opts.refit = options.refit
            rel_opts.sig_norm = options.sig_norm
            rel_opts.recover = options.recover
            if(len(options.effs) >0): rel_opts.effs = options.effs
            if('generic_sig_shape' in options.__dict__.keys()): rel_opts.generic_sig_shape = options.generic_sig_shape
            rel_opts.sig_shape = options.__dict__.get('sig_shape', "")
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
        spbs_to_run = sorted(options.spbs)


    #save options
    options.spbs = sorted(options.spbs)
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
    do_train =  "train" in options.step and 'sys' not in options.step
    do_clean = options.step == 'clean'
    get_condor = "get" in options.step  and 'sys' not in options.step
    do_selection = "select" in options.step and 'sys' not in options.step
    do_merge = "merge" in options.step and 'sys' not in options.step
    do_fit = "fit" in options.step
    do_plot = options.step == "plot"
    do_summary = options.step == "summary"
    do_roc = options.step == "roc"
    do_output = options.step == 'output'
    do_sys_train = "sys_train" in options.step
    do_sys_merge = options.step == "sys_merge"
    do_sys_selection = options.step == "sys_select"
    do_sys_get = "sys_get" in options.step or do_sys_selection
    do_sys_plot = options.step == "sys_plot"

    get_condor = get_condor or do_selection
    f_sig_effs = options.output + "sig_effs.npz"

    if(len(options.effs) == 0):
        eff_point = mass_bin_select_effs[options.mbin]
        options.effs = [eff_point]

    if('opt' in options.step):
        optimal_spb = get_optimal_spb(options)
        spbs_to_run = list(filter(lambda s : s >= optimal_spb, options.spbs))

    print(options.spbs)
    print("To run", spbs_to_run)


    #Do trainings
    if(do_train):
        for spb in spbs_to_run:
            t_opts = spb_opts(options, spb)
            t_opts.step = "train"
            #t_opts.condor = True
            t_opts.saved_AE_scores = True
            t_opts.max_events = 2000000
            t_opts.val_max_events = 200000
            if(t_opts.mbin % 10 < 3): t_opts.num_epoch = 50
            full_run(t_opts)
        
    if(get_condor):
        for spb in spbs_to_run:
            t_opts = spb_opts(options, spb)
            t_opts.step = "get"
            t_opts.reload = True
            t_opts.condor = True
            full_run(t_opts)


    if(do_selection):

        print("Selecting with eff %.03f based on mbin %i" % (options.effs[0], options.mbin))

        for spb in spbs_to_run:
            print(spb)
            t_opts = spb_opts(options, spb)
            t_opts.step = "select"
            t_opts.reload = True
            t_opts.condor = options.condor
            #t_opts.condor = True
            full_run(t_opts)



    if(do_merge):
        for spb in options.spbs:
            t_opts = spb_opts(options, spb)
            t_opts.step = "merge"
            t_opts.reload = True
            t_opts.condor = True
            full_run(t_opts)


    #Clean 
    if(do_clean):
        for spb in options.spbs:
            t_opts = spb_opts(options, spb)
            t_opts.step = "clean"
            full_run(t_opts)

        print(options.saved_params.keys())
        if('sys_select_list' in options.saved_params.keys()):
            trained = options.saved_params['sys_select_list'] + ['rand%i' %i for i in range(num_rand)]
            for sys in trained:
                print(sys)
                s_opts = spb_opts(options, spb, sys = sys)
                s_opts.step = "clean"
                full_run(s_opts)



    if(do_fit):
        for spb in spbs_to_run:
            t_opts = spb_opts(options, spb)
            t_opts.step = "fit"

            t_opts.reload = True
            t_opts.condor = True

            #run with specific shape
            #t_opts.fit_label = "sig_shape_"
            #t_opts.generic_sig_shape = False
            full_run(t_opts)
            os.system("mv %s %s" % (t_opts.output + "fit_results_%.1f.json" % options.mjj_sig, t_opts.output + "fit_results_sig_shape_%.1f.json" % options.mjj_sig))




    if(do_roc):
        for spb in spbs_to_run:
            t_opts = spb_opts(options, spb)
            t_opts.step = "roc"
            t_opts.reload = True
            t_opts.condor = False
            full_run(t_opts)


    if(do_plot):
        if(not os.path.exists(os.path.join(options.output, "plots/"))):
            os.system("mkdir " + os.path.join(options.output, "plots/"))
        sig_effs = []
        signifs = []
        pvals = []
        get_signal_params(options)

        for spb in spbs_to_run:
            sig_eff = float(get_sig_eff(options.output + "spb" + str(float(spb)) + "/", eff = options.effs[0]))
            fit_res = get_fit_results(outdir = options.output + "spb" + str(float(spb)) + "/",  m = options.mjj_sig)
            if(fit_res is not None): 
                signif = float(fit_res.signif)
                pval = float(fit_res.pval)
            else: 
                pval = signif = 0.


            sig_effs.append(sig_eff)
            signifs.append(signif)
            pvals.append(pval)

        print("Sig Effs: ",  sig_effs)
        print("Significances " , signifs)

        if(not options.data):
            #Use expected limit from b-only fit
            fit_results = get_fit_results(options = options, m=options.mjj_sig)
            n_evts_exc = fit_results.exp_lim_events
        else:
            fit_results = get_data_fit_results(options)
            n_evts_exc = fit_results.obs_lim_events

        options.saved_params['n_evts_exc_obs'] = fit_results.obs_lim_events
        options.saved_params['n_evts_exc_exp'] = fit_results.exp_lim_events
        options.saved_params['n_evts_exc_exp_1sig_high'] = fit_results.exp_lim_1sig_high
        options.saved_params['n_evts_exc_exp_1sig_low'] = fit_results.exp_lim_1sig_low
        options.saved_params['fit_bonly'] = fit_results.__dict__


        options.saved_params['spbs'] = spbs_to_run
        options.saved_params['signifs'] = signifs
        options.saved_params['pvals'] = pvals
        options.saved_params['sig_effs'] = sig_effs

        sig_effs = np.array(sig_effs)
        np.savez(f_sig_effs, sig_eff = sig_effs)

        make_limit_plot(options, sig_effs, spbs_to_run)
        options.saved_params['preselection_eff'] = options.preselection_eff

        make_sig_eff_plot(options, sig_effs, spbs_to_run)


        if(np.sum(signifs) > 0):
            make_signif_plot(options, signifs, spbs_to_run)

    if(do_output):
        output_json(options)


    if(do_sys_train):
        if(len(spbs_to_run) == 1):
            inj_spb = spbs_to_run[0]
            print("Using spb %i" % inj_spb)
        else:
            inj_spb = get_optimal_spb(options)

        if(options.saved_params['xsec_exc_exp'] > options.saved_params['inc_xsec_exc_exp']): #don't retrain systematics if worse than inclusive
            print("Expected limit (%.2f) is worse than inclusive (%.2f), skipping systematics training" % ( options.saved_params['xsec_exc_exp'], options.saved_params['inc_xsec_exc_exp']))
            sys_train_list = []
            options.saved_params['sys_train_list'] = []


        else:
            if(options.sys_train_all): sys_train_list = sys_list
            elif( 'sys_train_list' in options.saved_params.keys() and not options.retrain):
                sys_train_list = options.saved_params['sys_train_list']
            else: 
                t_opts = spb_opts(options, inj_spb)
                t_opts.output = options.output  + "sig_sys_plots/"
                #t_opts.lund_weights = False
                if(options.do_TNT): t_opts.randsort = True
                t_opts.saved_AE_scores = True
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
                full_run(t_opts)


            #do rand trainings too
            for seed in range(num_rand):
                t_opts = spb_opts(options, inj_spb, sys = "rand%i" % seed)
                t_opts.BB_seed = seed
                t_opts.step = "train"
                t_opts.condor = True
                full_run(t_opts)


    if(do_sys_get):
        inj_spb = get_optimal_spb(options)
        trained = options.saved_params['sys_train_list']  
        if(os.path.exists(options.output + "rand0/")): trained += ['rand%i' %i for i in range(num_rand)]
        for sys in trained:
            t_opts = spb_opts(options, inj_spb, sys = sys)
            t_opts.step = "get"
            t_opts.condor = True
            full_run(t_opts)
        


    if(do_sys_selection): #compute eff for systematics

        if(len(spbs_to_run) == 1):
            inj_spb = spbs_to_run[0]
            print("Using spb %i" % inj_spb)
        else:
            inj_spb = get_optimal_spb(options)

        #add JME vars but remove duplicates
        sys_select_list = list(set(options.saved_params['sys_train_list'] + list(JME_vars)))

        options.saved_params['sys_select_list'] = sys_select_list



        t_opts_orig = spb_opts(options, inj_spb)
        for sys in sys_select_list:
            t_opts = spb_opts(options, inj_spb, sys = sys)
            if(sys in options.saved_params['sys_train_list']):
                t_opts.reload = True
                #continue
            else:
                t_opts.reload = False
                t_opts.new = True
                t_opts.condor_mem = options.condor_mem
                #for now copy everything, TODO make more memory efficienct using sym links for models
                print_and_do("rm -r %s; cp -r %s %s" % (t_opts.output, t_opts_orig.output, t_opts.output))
                
            t_opts.step = "select"
            t_opts.eff_only = True
            t_opts.condor = True
            full_run(t_opts)

        if(options.saved_params['xsec_exc_exp'] < options.saved_params['inc_xsec_exc_exp']): #systematics not trained if worse than inclusive
            for seed in range(num_rand):
                #continue
                t_opts = spb_opts(options, inj_spb, sys = "rand%i" % seed)
                t_opts.step = "select"
                t_opts.eff_only = True
                t_opts.reload = True
                #t_opts.new = True
                t_opts.condor = True
                full_run(t_opts)

    if(do_sys_merge):
        if(len(spbs_to_run) == 1):
            inj_spb = spbs_to_run[0]
            print("Using spb %i" % inj_spb)
        else:
            inj_spb = get_optimal_spb(options)

        sys_select_list = options.saved_params["sys_select_list"]

        for sys in sys_select_list:
            print(sys)
            t_opts = spb_opts(options, inj_spb, sys = sys)
            t_opts.step = "merge"
            t_opts.eff_only = True
            t_opts.reload = True
            #t_opts.new = True
            t_opts.condor = True
            full_run(t_opts)

        if(options.saved_params['xsec_exc_exp'] < options.saved_params['inc_xsec_exc_exp']): #systematics not trained if worse than inclusive
            for seed in range(num_rand):
                t_opts = spb_opts(options, inj_spb, sys = "rand%i" % seed)
                t_opts.step = "merge"
                t_opts.eff_only = True
                t_opts.reload = True
                t_opts.condor = True
                full_run(t_opts)

    if(do_sys_plot):

        sig_effs = []
        for spb in options.spbs:
            sig_eff = get_sig_eff(options.output + "spb" + str(float(spb)) + "/", eff = options.effs[0])
            sig_effs.append(sig_eff)

        if(len(spbs_to_run) == 1):
            inj_spb = spbs_to_run[0]
        else:
            inj_spb = get_optimal_spb(options)

        print("Using spb %i" % inj_spb)


        sig_eff_nom = sig_eff_nom_fixed = sig_effs[options.spbs.index(inj_spb)]
        sig_eff_file = options.output + "spb" + str(float(inj_spb)) + "/"
        #dictionary of change in eff
        diffs_dict_retrain = dict() #when retrained
        diffs_dict_fixed = dict() #for fixed training
        diffs_dict_final = dict() #final combo
        #initialize
        for sys_clean in all_sys_list_clean:
            diffs_dict_retrain[sys_clean] = [0., 0.] 
            diffs_dict_fixed[sys_clean] = [0., 0.] 

        #random variation
        if(options.saved_params['xsec_exc_exp'] < options.saved_params['inc_xsec_exc_exp']): #systematics not trained if worse than inclusive
            rand_effs = [sig_eff_nom]
            #rand_effs = []
            for seed in range(num_rand):
                eff = get_sig_eff(options.output + "rand%i" % seed  + "/", eff = options.effs[0])
                rand_effs.append(eff)

            #sig_eff_nom = np.median(rand_effs)
            sig_eff_nom = np.mean(rand_effs)
            print(rand_effs)
            print("Nominal eff is %.4f " % sig_eff_nom)
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
                    sys_eff_fixed = get_sig_eff(options.output + sys + "_fixed/", eff = options.effs[0], noprint = True)
                else: 
                    sys_eff_fixed = eff
            else:
                if(sys_retrained): sys_eff_retrain = get_sig_eff(options.output + sys + "/", eff = options.effs[0])
                sys_eff_fixed = get_sig_eff(sig_eff_file, eff = options.effs[0], sys = sys)


            if(sys_eff_retrain > 0.): 
                #Jet mass variations use same random sampling as nominal, just slightly shifted, so compare to it to reduce stat fluctuations
                if('JM' in sys): diff_retrain =  sys_eff_retrain - sig_eff_nom_fixed
                #otherwise compare to avg of random variations
                else: diff_retrain =  sys_eff_retrain - sig_eff_nom
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

        #do special lund uncs
        lund_match_unc = get_matching_unc(options.sig_file)
        lund_stat_var_effs = get_sig_eff(sig_eff_file, eff = options.effs[0], sys = 'lund_stat')
        lund_pt_var_effs = get_sig_eff(sig_eff_file, eff = options.effs[0], sys = 'lund_pt')
        
        lund_stat_mean = np.mean(lund_stat_var_effs)
        lund_stat_std = np.std(lund_stat_var_effs)
        lund_pt_mean = np.mean(lund_pt_var_effs)
        lund_pt_std = np.std(lund_pt_var_effs)

        lund_stat_unc = (abs(lund_stat_mean - sig_eff_nom_fixed)  + lund_stat_std)
        lund_pt_unc = (abs(lund_pt_mean - sig_eff_nom_fixed) + lund_pt_std)
        diffs_dict_fixed['lund_match'][0] =  lund_match_unc * sig_eff_nom
        diffs_dict_fixed['lund_match'][1] = -lund_match_unc * sig_eff_nom

        diffs_dict_fixed['lund_stat'][0] =  lund_stat_unc
        diffs_dict_fixed['lund_stat'][1] = -lund_stat_unc
        diffs_dict_fixed['lund_pt'][0] =  lund_pt_unc
        diffs_dict_fixed['lund_pt'][1] = -lund_pt_unc


            
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
                #if(abs(diffs_dict_retrain[key][0]) > abs(diffs_dict_fixed[key][0])): up = diffs_dict_retrain[key][0]
                #if(abs(diffs_dict_retrain[key][1]) > abs(diffs_dict_fixed[key][1])): down = diffs_dict_retrain[key][1]
                up = diffs_dict_retrain[key][0]
                down = diffs_dict_retrain[key][1]

                diffs_dict_final[key] = [up, down]
            else:
                diffs_dict_final[key] = diffs_dict_fixed[key]
            #if("JER" in key): diffs_dict_final[key] = [0,0]

        #1.6% https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun2
        lumi_unc = 0.016
        diffs_dict_final['lumi'] = [lumi_unc * sig_eff_nom, -lumi_unc * sig_eff_nom]

        diffs_final_sorted = sorted(diffs_dict_final.items(), key = sort_fn) 
        make_sys_plot(options, sig_eff_nom, diffs_final_sorted, extra_label = "_final")

        up_tot = down_tot = 0.
        for key,vals in diffs_final_sorted:
            up_err = max(vals[0], vals[1])
            down_err = min(vals[0], vals[1])
            if(up_err > 0): up_tot += up_err**2
            if(down_err < 0): down_tot += down_err**2
            if(abs(up_err) > 0.01 or abs(down_err) > 0.01):
                print(key, up_err, down_err)

        up_tot = up_tot ** 0.5
        down_tot = down_tot ** 0.5

        print(" \n Eff final %.3f + %.3f - %.3f \n" % (sig_eff_nom, up_tot, down_tot))
        options.saved_params['sig_eff_nom'] = float(sig_eff_nom)
        options.saved_params['sig_eff_up_unc'] = float(up_tot)
        options.saved_params['sig_eff_up_down'] = float(down_tot)


        frac_unc = (up_tot + down_tot) / (2. * sig_eff_nom)
        #frac_unc = 0.0001

        #t_opts = spb_opts(options, inj_spb)
        #t_opts.step = "fit"
        #t_opts.fit_label = 'sys_final'
        #t_opts.sig_norm_unc = frac_unc
        #t_opts.reload = False
        #t_opts.condor = False
        #t_opts.generic_sig_shape = False

        #new_fit_loc = options.output + "data_fit_results_sys.json"
        #if(os.path.exists(new_fit_loc)):
        #    print("Prior fit found, not rerunning sig norm unc fit")
        #else:
        #    full_run(t_opts)
        #    fit_file = t_opts.output + 'fit_results_%.1f.json' % options.mjj_sig
        #    os.system("cp %s %s"  % (fit_file, new_fit_loc))

        if(not options.data):
            #Use expected limit from b-only fit
            fit_results = get_fit_results(options = options, m=options.mjj_sig)
            n_evts_exc = fit_results.exp_lim_events
        else:
            fit_results = get_data_fit_results(options, sig_norm_unc = frac_unc)

        get_signal_params(options)

        injected_xsec = inj_spb*options.numBatches / options.lumi / options.preselection_eff / options.hadronic_only_eff

        options.saved_params['n_evts_exc_obs_sys'] = fit_results.obs_lim_events
        options.saved_params['n_evts_exc_exp_sys'] = fit_results.exp_lim_events
        options.saved_params['n_evts_exc_exp_1sig_high_sys'] = fit_results.exp_lim_1sig_high
        options.saved_params['n_evts_exc_exp_1sig_low_sys'] = fit_results.exp_lim_1sig_low

        if(options.saved_params['best_exp_spb'] == options.saved_params['best_spb']):
            sig_eff_nom_exp = sig_eff_nom
            inj_exp = injected_xsec
        else:
            sig_eff_nom_exp = sig_effs[options.spbs.index(options.saved_params['best_exp_spb'])]
            inj_exp = options.saved_params['best_exp_spb']*options.numBatches / options.lumi / options.preselection_eff / options.hadronic_only_eff

        #obs used to be default
        if('best_obs_spb' not in options.saved_params.keys() or options.saved_params['best_obs_spb'] == options.saved_params['best_spb']):
            sig_eff_nom_obs = sig_eff_nom
            inj_obs = injected_xsec
        else:
            sig_eff_nom_obs = sig_effs[options.spbs.index(options.saved_params['best_obs_spb'])]
            inj_obs = options.saved_params['best_obs_spb']*options.numBatches / options.lumi / options.preselection_eff / options.hadronic_only_eff


        obs_excluded_xsec = max(inj_obs, convert_to_xsec(options, options.saved_params['n_evts_exc_obs_sys'], sig_eff_nom_obs))
        exp_excluded_xsec = max(inj_exp, convert_to_xsec(options, options.saved_params['n_evts_exc_exp_sys'], sig_eff_nom_exp))
        exp_excluded_xsec_1sig_high = max(inj_exp, convert_to_xsec(options, options.saved_params['n_evts_exc_exp_1sig_high_sys'], sig_eff_nom_exp))
        exp_excluded_xsec_1sig_low = max(inj_exp, convert_to_xsec(options, options.saved_params['n_evts_exc_exp_1sig_low_sys'], sig_eff_nom_exp))

        print("\nInclusive obs %.2f, exp %.2f" % (options.saved_params['inc_xsec_exc_obs'], options.saved_params['inc_xsec_exc_exp']))
        print("No sys: obs %.2f, exp %.2f" % (options.saved_params['xsec_exc_obs'], options.saved_params['xsec_exc_exp']))
        print("Final result: Injected %.2f Obs excluded %.2f. Exp excluded %.2f + %.2f - %.2f\n" % 
                (injected_xsec, obs_excluded_xsec, exp_excluded_xsec, exp_excluded_xsec_1sig_high, exp_excluded_xsec_1sig_low))


        options.saved_params['xsec_exc_obs_sys'] = obs_excluded_xsec
        options.saved_params['xsec_exc_exp_sys'] = exp_excluded_xsec
        options.saved_params['xsec_exc_exp_1sig_high_sys'] = exp_excluded_xsec_1sig_high
        options.saved_params['xsec_exc_exp_1sig_low_sys'] = exp_excluded_xsec_1sig_low


    write_params(options.output + "saved_params.json", options.saved_params)




if(__name__ == "__main__"):
    parser = input_options()
    parser.add_argument("--sig_norm", default = -1, type = int,   help = 'Signal normalization for fit')
    parser.add_argument("--sig_norm_unc", default = -1.0, type = float, help = "parameter for fit (uncertainty on signal efficiency)")
    parser.add_argument("--ae_dir", default = "", help = "directory with all the autoencoders (auto pick the right mbin and kfold)")
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
    parser.add_argument("--sig_shape", default = "", help='signal shape file')
    parser.add_argument("--generic_sig_shape", default = False, action ='store_true')
    parser.add_argument("--no_generic_sig_shape", dest = 'generic_sig_shape', action ='store_false')
    parser.set_defaults(reload=True)
    parser.set_defaults(deta=1.3)
    parser.add_argument("--condor", dest = 'condor', action = 'store_true')
    parser.add_argument("--condor_mem", default = -1, type = int, help = "Memory for condor jobs")
    parser.add_argument("--no-condor", dest = 'condor', action = 'store_false')
    parser.add_argument("--recover", default = False, dest='recover', action = 'store_true', help = "Retrain only missing jobs")
    parser.add_argument("--retrain", default = False, dest='retrain', action = 'store_true', help = "Retrain jobs that failed")
    parser.add_argument("--refit", action = 'store_true', help = 'redo no selection signal fit')
    parser.set_defaults(condor=True)
    parser.set_defaults(num_models=3)
    options = parser.parse_args()
    limit_set(options)
