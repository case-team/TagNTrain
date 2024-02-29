import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import mplhep as hep
import scipy.integrate as integrate
import scipy, scipy.stats
import h5py
import uncertainties
from uncertainties import unumpy as unp
import argparse
import json
import os
import copy


def get_rebinning(xbins, mjjs, min_count = 5):

    rebins = copy.deepcopy(xbins)
    below_min = True

    while(below_min):
        vals, edges = np.histogram(mjjs, bins=rebins)
        below_bins = np.argwhere(vals < min_count)
        if(len(below_bins) < 1): below_min = False
        else: rebins = np.delete(rebins, below_bins[-1])


    return rebins


def doubleCB(x, mu, sigma, alpha1, n1, alpha2, n2):
    z = (x - mu) / sigma

    A1 = (n1 / abs(alpha1)) ** n1 * np.exp(-alpha1**2/2)
    A2 = (n2 / abs(alpha2)) ** n2 * np.exp(-alpha2**2/2)
    B1 = n1 / abs(alpha1) - np.abs(alpha1)
    B2 = n2 / abs(alpha2) - np.abs(alpha2)

    #norm factor
    N = 1 

    if(z < -alpha1):
        return N * A1 * (B1 - z)**(-n1)
    elif(z < alpha2):
        return N * np.exp(-z**2/2)
    else:
        return N * A2 * (B2 + z)**(-n2)



#@uncertainties.wrap
def qcd_model(m, p1, p2, p3=0, p4=0):
    x = m / 13000.
    y_vals = ((1-x)**p1)/(x**(p2+p3*np.log(x)+p4*np.log(x)**2))
    return y_vals

@uncertainties.wrap
def integrate_qcd_model(a,b, p1,p2,p3=0,p4=0):
    integral, error = integrate.quad(qcd_model, a=a,b=b, args = (p1,p2,p3,p4))
    return integral 


def plot_mjj_fit(options):
    num_div = 75
    bin_size_scale = 100.



    plt.style.use(hep.style.CMS)
    a1 = plt.axes([0.0, 0.18, 1.0, 0.8])
    a2 = plt. axes([0.0,0.0, 1.0, 0.16], sharex=a1)
    plt.sca(a1)
    #hep.cms.label(llabel = " Preliminary", lumi = 138)

    hep.cms.text("Preliminary")
    hep.cms.lumitext("138 fb$^{-1}$ (13 TeV)")

    plt.sca(a1)

    f_data = options.fin

    with h5py.File(f_data, "r") as f:
        mjjs = f['mjj'][()]


    fit_file = options.fit_file
    print(fit_file)
    if(not os.path.exists(fit_file)): 
        print("Missing " + fit_file)
        exit(1)

    with open(fit_file) as json_file:
        results = json.load(json_file)

    xbins = np.array([1460, 1530, 1607, 1687, 1770, 1856, 1945, 2037, 2132, 2231, 2332, 2438,
             2546, 2659, 2775, 2895, 3019, 3147, 3279, 3416, 3558, 3704, 3854,
             4010, 4171, 4337, 4509, 4700, 4900,  5100, 5300, 5500, 5800,
             6100, 6500])


    #rebinning to match dijetfit code
    bins_nonzero = get_rebinning(xbins, mjjs)
    print("Rebinning to avoid zero bins!")
    print("old", xbins)
    print("new", bins_nonzero)
    xbins = bins_nonzero

    mlow = xbins[0]
    mhigh = xbins[-1]

    nbins_fine = 500
    xbins_fine = np.linspace(mlow, mhigh, nbins_fine + 1)

    vals, edges = np.histogram(mjjs, bins=xbins)
    widths = np.diff(xbins)
    centers = (xbins[:-1] + xbins[1:])/2
    centers_fine = (xbins_fine[:-1] + xbins_fine[1:])/2
    widths_fine = np.diff(xbins_fine)

    vals_norm = vals * (bin_size_scale / widths)
    errs_norm = np.sqrt(vals) * ( bin_size_scale/ widths)

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

    fit_mjj_start = max(1450, results['mjj_min'])
    fit_mjj_stop = min(6500, results['mjj_max'])
    mjj_fit = mjjs[ (mjjs > fit_mjj_start) & (mjjs < fit_mjj_stop )]
    n_evts_fit = mjj_fit.shape[0]


    #normalize fit integral to total number of data events
    fit_norm = n_evts_fit / integrate_qcd_model(fit_mjj_start, fit_mjj_stop, *pars)

    #integrate fit pdf in each bin to get predictions
    fit_vals_fine = np.array([fit_norm * integrate_qcd_model(xbins_fine[k], xbins_fine[k+1], *pars) for k in range(len(xbins_fine)-1)])


    #for ratio panel use big bins
    fit_vals = np.array([fit_norm * integrate_qcd_model(edges[k], edges[k+1], *pars) for k in range(len(edges)-1)])
    fit_errs = unp.std_devs(fit_vals) * (bin_size_scale  / widths)
    fit_nom = unp.nominal_values(fit_vals) * (bin_size_scale  / widths)

    #for upper panel use fine binning
    fit_fine_nom = unp.nominal_values(fit_vals_fine) * (bin_size_scale / widths_fine)
    fit_fine_unc = unp.std_devs(fit_vals_fine) * (bin_size_scale / widths_fine)
    fit_fine_up = fit_fine_nom + fit_fine_unc
    fit_fine_down = fit_fine_nom - fit_fine_unc


    label_fit, label_data = "Bkg. fit", "Data"
    plt.fill_between(centers_fine, fit_fine_down, fit_fine_up, color = 'red', linewidth=0, alpha = 0.5)
    plt.plot(centers_fine, fit_fine_nom, color = 'red', linewidth=3, label = label_fit) 

    plt.errorbar(centers, vals_norm, fmt="ko", xerr=xbins[1:]-centers, yerr=errs_norm,  elinewidth=0.5, capsize=1.5, label = label_data)

    draw_sig_shapes = True
    if(draw_sig_shapes):
        # 3TeV XtoYY
        mu = 3000.
        sigma = 110.3
        alpha1 = 1.8
        alpha2 = 1.83
        n1 = 4.6
        n2 = 3.3

        plot_norm = 1e4

        norm_fac, error = integrate.quad(doubleCB, a=mlow,b=mhigh, args = (mu,sigma, alpha1, n1, alpha2, n2))

        sig_vals_fine = np.array([doubleCB(x, mu, sigma, alpha1, n1, alpha2, n2) for x in centers_fine])
        sig_vals_fine *= 1e4 / norm_fac


        filt = (centers_fine > 2500) & (centers_fine < 3500)

        plt.plot(centers_fine[filt], sig_vals_fine[filt], color = 'blue', linewidth=3, linestyle = 'dashed', label = r"3 TeV X$\to$YY'$\to$4q") 


        # 5TeV W'
        mu = 4771
        sigma = 262
        alpha1 = 0.72
        alpha2 = 1.95
        n1 = 0.82
        n2 = 3.18

        plot_norm = 1e2

        norm_fac, error = integrate.quad(doubleCB, a=mlow,b=mhigh, args = (mu,sigma, alpha1, n1, alpha2, n2))

        sig_vals_fine = np.array([doubleCB(x, mu, sigma, alpha1, n1, alpha2, n2) for x in centers_fine])
        sig_vals_fine *= 1e3 / norm_fac

        filt = (centers_fine > 4000) & (centers_fine < 5500)

        plt.plot(centers_fine[filt], sig_vals_fine[filt], color = 'purple', linewidth=3, linestyle = 'dashed', label = r"5 TeV W'$\to$B't$\to$bZt") 



    # ratio plot
    plt.sca(a2)
    
    # in first iteration, plot dashed line at zero
    max_ratio = 4
    min_ratio = -4
    plt.plot([mlow, mhigh], [0, 0], color="black", linestyle="dashed")
    
    tot_unc = np.sqrt(fit_errs**2 + errs_norm**2)
    pulls = (vals_norm-fit_nom)/tot_unc
    bottoms = np.copy(pulls)
    bottoms[bottoms > 0.0] = 0.0
    plt.hist(centers, bins=edges, weights=pulls, histtype="stepfilled", color="gray")


    plt.sca(a1)
    plt.ylabel("Events / 100 GeV", fontsize = 36)
    plt.yscale("log")
    ymin = np.amin(vals_norm)
    plt.ylim(0.1 * ymin, 1e5)

    handles, labels = plt.gca().get_legend_handles_labels()
    #reorder data to be first in legend
    order = [3,0,1,2] if draw_sig_shapes else [1,0]

    #add legend to plot
    legend = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = "upper right", fontsize = 'large',
            facecolor="white", framealpha=1, frameon=True,)
    legend.get_frame().set_linewidth(0)

    method_label = options.label
    plt.text(0.3, 0.9, method_label , horizontalalignment = 'center', transform = a1.transAxes)



    #plt.legend(loc="upper right")


    plt.sca(a2)
    plt.xlabel(r"$m_{jj}$ (GeV)", fontsize = 36)
    plt.ylabel(r"$\frac{\mathrm{Data-Fit}}{\mathrm{Unc.}}$", fontsize=30)

    plt.ylim(min_ratio, max_ratio)
    plt.xlim(mlow, mhigh)
    tick_spacing = 2.0
    a2.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    a2.yaxis.set_minor_locator(ticker.MultipleLocator(tick_spacing/2))
    a1.tick_params(axis='x', labelbottom=False)

    plt.savefig(options.output, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--fin", default='', help="Input file with Mjjs")
    parser.add_argument("-l", "--label", default='', help="plot label")
    parser.add_argument("-f", "--fit_file", default='', help="Fit results (json)")
    parser.add_argument("-o", "--output", default='', help="Output file")
    options = parser.parse_args()
    plot_mjj_fit(options)
