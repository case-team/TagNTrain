
from full_run import *

from matplotlib import pyplot as plt
import mplhep as hep
from os.path import join
from scipy.special import erf
import scipy.integrate as integrate
import matplotlib
import uncertainties
from uncertainties import unumpy as unp


def qcd_model(m, p1, p2, p3=0, p4=0):
    x = m / 13000.
    y_vals = ((1-x)**p1)/(x**(p2+p3*np.log(x)+p4*np.log(x)**2))
    return y_vals


def plot_biases(sig_masses, mean_pulls, err_mean_pulls, outfile):
    plt.style.use(hep.style.CMS)
    plt.errorbar(sig_masses, mean_pulls[0], yerr = err_mean_pulls[0], label = "0 Sigma Inj.", c = "black", fmt = "o")
    plt.errorbar(sig_masses, mean_pulls[1], yerr = err_mean_pulls[1], label = "2 Sigma Inj.",  c = "green", fmt = "o")
    plt.errorbar(sig_masses, mean_pulls[2], yerr = err_mean_pulls[2], label = "5 Sigma Inj.",  c = "blue",  fmt = "o")

    plt.legend(loc='upper right')
    plt.xlabel(r"$m_{jj}$ [GeV]")
    plt.ylabel(r"Mean Bias $(\frac{\mu_{fit} - \mu_{gen}}{\sigma_{\mu}})$")
    plt.ylim(-2.0, 2.0)

    #draw_mbins(plt, ymin = -1.5, ymax = 1.5, colors = ('gray', 'gray'))
    xmax = np.amax(sig_masses)
    plt.xlim(1400, xmax + 150.)
    hep.cms.text("Preliminary")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    print("Wrote out %s" % outfile)
    plt.close()


def plot_stitched_mjj(options, mbins, outfile):
    if(mbins[0] < 10):
        mass_bins = mass_bins1
    else:
        mass_bins = mass_bins2

    mjj_min = mass_bins[1]
    mjj_max = mass_bins[-2]
    num_div = 75
    bin_size_scale = 100.


    plt.style.use(hep.style.CMS)
    a1 = plt.axes([0.0, 0.18, 1.0, 0.8])
    a2 = plt. axes([0.0,0.0, 1.0, 0.16], sharex=a1)
    plt.sca(a1)
    #hep.cms.text("Simulation, Work in progress")

    for mbin in mbins:
        plt.sca(a1)

        t_opts = mbin_opts(options, mbin)
        sig_mass  = mass_bin_sig_mass_map[mbin][0]

        f_data = t_opts.output + "fit_inputs_eff%.1f.h5" % mass_bin_select_effs[mbin]

        with h5py.File(f_data, "r") as f:
            mjjs = f['mjj'][()]

        mjjs = mjjs[ (mjjs > mjj_min) & (mjjs < mjj_max)]
        n_evts = mjjs.shape[0]
        print("n_evts", n_evts)

        fit_file = t_opts.output + 'fit_results_%.1f.json' % sig_mass
        if(not os.path.exists(fit_file)): 
            print("Missing " + fit_file)
            exit(1)

        with open(fit_file) as json_file:
            results = json.load(json_file)


        sr_mlow = mass_bins[mbin % 10]
        sr_mhigh = mass_bins[ (mbin+1) % 10]


        #tmp_mjj = mjjs[(mjjs > mjj_min) & (mjjs < mjj_max)]
        #xbins = np.linspace(mjj_min, mjj_max, 20)
        tmp_mjj = mjjs[(mjjs > sr_mlow) & (mjjs < sr_mhigh)]
        xbins = np.linspace(sr_mlow, sr_mhigh, 5)

        vals, edges = np.histogram(tmp_mjj, bins=xbins)
        widths = edges[1:] - edges[:-1]
        centers = (edges[1:] + edges[:-1])/2

        #vals_norm = vals / (widths/ bin_size_scale)
        #errs_norm = np.sqrt(vals) / (widths/ bin_size_scale)
        vals_norm = vals
        errs_norm = np.sqrt(vals) 

        label_data = 'data'
        plt.errorbar(centers,vals_norm, yerr=errs_norm, label=label_data, fmt="ko")


        npars = results['nPars_QCD']
        fit_params = results['bkg_fit_params']
        p1 = p2 = p3 = p4 = 0
        cov = np.array(fit_params['cov'], dtype=np.float32)
        p1 = fit_params['p1'][0]
        p2 = fit_params['p2'][0]
        if(npars == 2): 
            pars = uncertainties.correlated_values([p1,p2], cov)
        elif(npars == 3): 
            p3 = fit_params['p3'][0]
            pars = uncertainties.correlated_values([p1,p2,p3], cov)
        elif(npars == 4): 
            p3 = fit_params['p3'][0]
            p4 = fit_params['p4'][0]
            pars = uncertainties.correlated_values([p1,p2,p3,p4], cov)

        mjj_fit = mjjs[(mjjs > mjj_min) & (mjjs < mjj_max)]
        n_evts_fit = mjj_fit.shape[0]


        #normalize fit integral to total number of data events
        fit_norm = n_evts_fit / integrate.quad(qcd_model, a=mjj_min, b=mjj_max, args = (p1,p2,p3,p4))[0]
        print("fit norm, n_evts_fit", fit_norm, n_evts_fit)

        #integrate fit pdf to get more accurate vals in each bin
        fit_nom = [fit_norm * integrate.quad(qcd_model, a= edges[k], b = edges[k+1], args = (p1,p2,p3,p4))[0] for k in range(len(edges)-1)]



        #get fractional error on fit based on bin center
        #TODO These fractional errors don't really match those reported by the fit... 
        fit_center_errs = qcd_model(centers, *pars)
        fit_center_errs /= np.sum(fit_center_errs)
        print(centers)
        print(fit_center_errs)
        fit_frac_err = unp.std_devs(fit_center_errs) / unp.nominal_values(fit_center_errs)
        print(fit_frac_err)

        fit_up = (1. + fit_frac_err) * fit_nom
        fit_down = (1. - fit_frac_err) * fit_nom
        fit_errs = (fit_up - fit_down) / 2.0


        label_fit = 'fit'
        plt.hist(centers, bins=edges, weights=fit_nom, histtype="step", label=label_fit, color="red") 
        plt.hist(centers, bins=edges, weights=(fit_up), histtype="step", color="red", linestyle="dashed")
        plt.hist(centers, bins=edges, weights=(fit_down), histtype="step", color="red", linestyle="dashed")

        
                        
        # ratio plot
        plt.sca(a2)
        
        # in first iteration, plot dashed line at zero
        if(mbin %10 == 1):
            plt.plot([1650.0, 5500.0], [0, 0], color="black", linestyle="dashed")
            max_ratio = 3
            min_ratio = -3
        else:
            plt.plot([edges[0], edges[0]], [min_ratio, max_ratio], color="lightgreen", linestyle="dashed", lw=1.0)
            plt.plot([edges[-1], edges[-1]], [min_ratio, max_ratio], color="lightgreen", linestyle="dashed", lw=1.0)
        
        tot_unc = np.sqrt(fit_errs**2 + errs_norm**2)
        pulls = (vals_norm-fit_nom)/tot_unc
        print(pulls)
        bottoms = np.copy(pulls)
        bottoms[bottoms > 0.0] = 0.0
        plt.hist(centers, bins=edges, weights=pulls, histtype="stepfilled", color="gray")


    plt.sca(a1)
    plt.ylabel("Events")
    plt.yscale("log")
    #plt.legend(loc="upper right")

    plt.sca(a2)
    plt.xlabel(r"$m_{jj}$ (GeV)")
    plt.ylabel(r"$\frac{\mathrm{Data-Fit}}{\sigma_{\mathrm{tot.}}}$")
    plt.ylim(min_ratio, max_ratio)
    plt.xlim(mjj_min, mjj_max)
    a1.tick_params(axis='x', labelbottom=False)

    plt.savefig(outfile, dpi=300, bbox_inches="tight")



def plot_significances(input_files, out_dir, sig_masses = None):

    # build arrays of variables to plot
    if(sig_masses is None):
        mass_list = []
    else:
        mass_list = sig_masses
    pval_list = []
    signif_list = []
    nPar_list = []
    fit_prob_list = []
    fit_start_list = []

    default_fit_start = 1450.

    for f in input_files:
        with open(f) as json_file:
            results = json.load(json_file)

        if(sig_masses is None): mass_list.append(results["mass"])
        pval_list.append(results["pval"])
        signif_list.append(results["signif"])
        nPar_list.append(results["nPars_QCD"])
        fit_prob_list.append(results["bkgfit_prob"])
        #map -1 to starting fit value
        fit_start_list.append(max(default_fit_start, results['mjj_min']))

    xmax = np.amax(mass_list)

    corr_sigma = []
    for i in range(1, 6):
        tmp = 0.5-(0.5*(1+erf(i/np.sqrt(2)))-0.5*(1+erf(0/np.sqrt(2))))
        #if i == 1:
            #print(f"This should be 0.1586553: {tmp}")
        corr_sigma.append(tmp)

    pval_overflow_list = []
    mass_pval_overflow_list = []
    for i,p in enumerate(pval_list):
        if( p < 1e-7):
            mass_pval_overflow_list.append(mass_list[i])
            pval_overflow_list.append(2e-7)

    mass_list = np.array(mass_list)
    pval_list = np.array(pval_list)
    mass_bins = [sig_mass_to_mbin(m) for m in mass_list]
    colors = ['blue' if mbin < 10 else 'green' for mbin in mass_bins]

    mass_pval_overflow_list = np.array(mass_pval_overflow_list)
    pval_overflow_list = np.array(pval_overflow_list)
    mass_bins_overflow = [sig_mass_to_mbin(m) for m in mass_pval_overflow_list]
    colors_overflow = ['blue' if mbin < 10 else 'green' for mbin in mass_pval_overflow_list]

    signif_list = np.array(signif_list)

    #pval plot
    plt.style.use(hep.style.CMS)
    plt.scatter(mass_list, pval_list, s = 40.0, c = colors)
    plt.scatter(mass_pval_overflow_list, pval_overflow_list, marker = "v", s = 80.0, c = colors_overflow)
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
    draw_mbins(plt, ymin = 1e-7, ymax = 10)
    plt.xlim(1400, xmax + 150.)
    hep.cms.text("Preliminary")
    plt.savefig(join(out_dir, "pval_plot.png"), dpi=300, bbox_inches="tight")
    plt.close()

    #signif plot
    plt.style.use(hep.style.CMS)
    hep.cms.text("Preliminary")
    plt.scatter(mass_list, signif_list, s = 40.0, c = colors)
    plt.xlabel(r"$m_{jj}$ [GeV]")
    plt.ylabel(r"Significance [$\sigma$]")
    plt.ylim(-0.2, None)
    draw_mbins(plt)
    plt.xlim(1400, xmax + 150.)
    plt.savefig(join(out_dir, "signif_plot.png"), dpi=300, bbox_inches="tight")
    plt.close()


    #nPar_qcd plot
    plt.style.use(hep.style.CMS)
    plt.scatter(mass_list, nPar_list, s = 40.0, c = colors)
    plt.xlabel(r"$m_{jj}$ [GeV]")
    plt.ylabel("nPars")
    draw_mbins(plt)
    plt.ylim(0., np.amax(nPar_list) + 0.5)
    plt.xlim(1400, xmax + 150.)
    plt.savefig(join(out_dir, "nPar_plot.png"), dpi=300, bbox_inches="tight")
    plt.close()


    #chi2 prob plot
    plt.style.use(hep.style.CMS)
    plt.scatter(mass_list, fit_prob_list, s = 40.0, c = colors)
    plt.xlabel(r"$m_{jj}$ [GeV]")
    plt.ylabel("Bkg. Only Fit Prob.")
    #plt.ylim(1e-8, 1.1)
    #plt.yscale('log')
    draw_mbins(plt)
    plt.ylim(-0.1, 1.1)
    plt.xlim(1400, xmax + 150.)
    plt.savefig(join(out_dir, "fit_prob_plot.png"), dpi=300, bbox_inches="tight")
    plt.close()


    #fit start plot
    plt.style.use(hep.style.CMS)
    plt.scatter(mass_list, fit_start_list, s = 40.0, c = colors)
    plt.xlabel(r"$m_{jj}$ [GeV]")
    plt.ylabel(r"Lowest $m_{jj}$ bin of fit")
    #plt.ylim(1e-8, 1.1)
    #plt.yscale('log')
    draw_mbins(plt)
    fit_start_max = np.amax(fit_start_list)
    ymax = max(2400., fit_start_max + 50.)
    plt.ylim(1300., ymax)
    plt.xlim(1400, xmax + 150.)
    plt.savefig(join(out_dir, "fit_start_plot.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("Done! Plots saved to %s" % os.path.abspath(out_dir))






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
            rel_opts.recover = options.recover
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
    do_bias_test = options.step == "bias"
    do_bias_plot = options.step == "bias_plot"

    get_condor = get_condor or do_selection
    

    #Do trainings
    sb_excluded_mbins = [1,11]
    #if(do_train):
    #    for mbin in mass_bin_idxs:
    #        if(options.sideband and mbin in sb_excluded_mbins): continue
    #        t_opts = mbin_opts(options, mbin)
    #        t_opts.step = "train"
    #        full_run(t_opts)

    if(do_train or do_selection or do_merge or get_condor):
        for mbin in mass_bin_idxs:
            if(options.sideband and mbin in sb_excluded_mbins): continue
            t_opts = mbin_opts(options, mbin)
            t_opts.step = options.step
            t_opts.condor = True
            if(not do_train): t_opts.reload = True
            full_run(t_opts)


    #if(do_merge):
    #    for mbin in mass_bin_idxs:
    #        if(options.sideband and mbin in sb_excluded_mbins): continue
    #        t_opts = mbin_opts(options, mbin)
    #        t_opts.step = "merge"
    #        t_opts.reload = True
    #        t_opts.condor = True
    #        full_run(t_opts)



    first_sb_sig_mass = 2250.
    if(do_fit):
        for mbin in mass_bin_idxs:
            if(options.sideband and mbin in sb_excluded_mbins): continue
            eff_point = mass_bin_select_effs[mbin]
            print("Mbin %i, eff %.1f" % (mbin, eff_point))
            t_opts = mbin_opts(options, mbin)
            t_opts.effs = [eff_point]
            t_opts.step = "fit"
            t_opts.reload = True
            t_opts.condor = False
            fit_start = -1
            if(options.sideband): t_opts.fit_start = 2000.
            for sig_mass in mass_bin_sig_mass_map[mbin]:
                if(options.sideband and sig_mass < first_sb_sig_mass): continue
                t_opts.mjj_sig = sig_mass
                print("mbin %i, sig_mass %.0f" %(mbin, sig_mass))
                fit_start = full_run(t_opts)
                #for signal masses in same mass bin save time by reusing fit start
                t_opts.fit_start = fit_start

    if(do_bias_test):
        for mbin in mass_bin_idxs:
            if(options.sideband and mbin in sb_excluded_mbins): continue
            eff_point = mass_bin_select_effs[mbin]
            print("Mbin %i, eff %.1f" % (mbin, eff_point))
            t_opts = mbin_opts(options, mbin)
            t_opts.effs = [eff_point]
            t_opts.step = "bias"
            t_opts.reload = True
            t_opts.condor = False
            fit_start = -1
            if(options.sideband): t_opts.fit_start = 2000.

            #Just do first sig mass per mbin for now
            for sig_mass in mass_bin_sig_mass_map[mbin]:
                if(options.sideband and sig_mass < first_sb_sig_mass): continue
                t_opts.mjj_sig = sig_mass
                print("mbin %i, sig_mass %.0f" %(mbin, sig_mass))
                fit_start = full_run(t_opts)

    if(do_bias_plot):
        sig_masses = []
        mean_pulls = [ [], [], [] ] 
        err_mean_pulls = [ [], [], [] ] 
        for mbin in mass_bin_idxs:
            t_opts = mbin_opts(options, mbin)
            for sig_mass in mass_bin_sig_mass_map[mbin]:

                sig_masses.append(sig_mass)

                for k,sigma in enumerate([0, 2, 5]):
                    results = t_opts.output  + "bias_test/" + 'bias_test_results_%.1f_%isigma.json' % (sig_mass, sigma)
                    pulls = []
                    with open(results, 'r') as f:
                        params = json.load(f, encoding="latin-1")
                        resids =  params['resids']
                        uncs = params['uncs']
                    for i in range(len(resids)):
                        #remove outliers from failed fits
                        if(abs(resids[i]/uncs[i]) < 5.): pulls.append(resids[i] / uncs[i])
                    mean_pull = np.mean(pulls)
                    std_pull = np.std(pulls)
                    mean_pulls[k].append(mean_pull)
                    err_mean_pulls[k].append(std_pull / np.sqrt(len(pulls)))

        print(mean_pulls[0])
        print(err_mean_pulls[0])

        plot_biases(sig_masses, mean_pulls, err_mean_pulls, options.output + "plots/bias_mean_pulls.png")

    if(do_plot):
        os.system("mkdir -p " + options.output + "plots/")
        sig_masses = []
        signifs = []
        pvals = []
        file_list = []
        file_list_bins1 = []
        file_list_bins2 = []
        for mbin in mass_bin_idxs:
            if(options.sideband and mbin in sb_excluded_mbins): continue
            t_opts = mbin_opts(options, mbin)

            fit_plot_dir = t_opts.output + "fit_plots_mbin%i/" % mbin
            os.system('mkdir %s; mv %s/*.png %s' % (fit_plot_dir, t_opts.output, fit_plot_dir))

            for sig_mass in mass_bin_sig_mass_map[mbin]:
                fit_plot = fit_plot_dir + 'sbFit_M%.0f_raw.png' % sig_mass
                if(options.sideband and sig_mass < first_sb_sig_mass): continue

                os.system('cp ' + fit_plot + ' %s/plots/sbfit_mbin%i_mjj%.0f.png' % (options.output, mbin, sig_mass))

                fit_file = t_opts.output + 'fit_results_%.1f.json' % sig_mass
                if(os.path.exists(fit_file)):
                    file_list.append(fit_file)
                    sig_masses.append(sig_mass)


                    with open(fit_file, 'r') as f:
                        fit_params = json.load(f, encoding="latin-1")
                        signifs.append(fit_params['signif'])
                        pvals.append(fit_params['pval'])
                        #should be the same for all signal masses in the mass bin
                        nPars_bkg = fit_params["nPars_QCD"]
                else:
                    print("Missing %s" % fit_file)

            bkg_fit_plot = fit_plot_dir + '%ipar_qcd_fit_binned.png' % nPars_bkg
            os.system('cp ' + bkg_fit_plot + ' %s/plots/bkgfit_mbin%i.png' % (options.output, mbin))

        
        print(list(zip(sig_masses, signifs)))
        plot_significances(file_list, options.output + "plots/", sig_masses = sig_masses)
        plot_stitched_mjj( options, [mbin for mbin in mass_bin_idxs if mbin < 10] , options.output + "plots/mjj_stitched_binsA.png")
        plot_stitched_mjj( options, [mbin for mbin in mass_bin_idxs if mbin > 10] , options.output + "plots/mjj_stitched_binsB.png")






            

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
    parser.add_argument("--recover", dest='recover', action = 'store_true', help = "Retrain jobs that failed")
    parser.add_argument("--new", dest='reload', action = 'store_false', help = "Reload based on previously saved options")
    parser.set_defaults(reload=True)
    parser.add_argument("--condor", dest = 'condor', action = 'store_true')
    parser.add_argument("--no-condor", dest = 'condor', action = 'store_false')
    parser.set_defaults(num_models=3)
    parser.set_defaults(condor=True)
    options = parser.parse_args()
    full_scan(options)
