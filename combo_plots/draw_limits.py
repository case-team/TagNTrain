import h5py
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.special import erf
import argparse
import sys
sys.path.append('..')
from utils.TrainingUtils import *
import mplhep as hep
import matplotlib
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

def make_limit_plot(vals, labels, colors, fout = ""):
    plt.style.use(hep.style.CMS)
    plt.figure(figsize=(12,9))


    legend_elems = []
                   

    for method_i,method_vals in enumerate(vals):
        for sig_idx in range(len(method_vals)):
            exp_lim, exp_lim_up, exp_lim_down, obs_lim = method_vals[sig_idx]
            x = (sig_idx + 1.0) + (method_i * 0.10) - 0.3
            err_up = exp_lim_up - exp_lim
            err_down = exp_lim - exp_lim_down
            label = labels[method_i] if sig_idx == 0 else ""
            exp = plt.errorbar(x, exp_lim, yerr = [[err_up], [err_down]], fmt = 'x', color = colors[method_i], label = label, capsize = 2.0) 
            obs = plt.plot(x, obs_lim, linewidth = 0, marker = 's', color = colors[method_i])[0]
            if(sig_idx == 0):
                legend_elems.append((exp,obs))

    plt.xlabel("Signal Model", labelpad =20)
    plt.ylabel("95% CL Exclusion Limit on $\sigma$ (fb)")
    plt.gca().minorticks_off()
    y_minor = matplotlib.ticker.MultipleLocator(1)
    plt.gca().yaxis.set_minor_locator(y_minor)
    ymax = np.amax(vals)
    plt.ylim(0, 1.5*ymax)
    plt.xlim(0.5, 5.5)
    plt.xticks([1,2,3,4,5], [r"$X \rightarrow YY'$" "\n(2+2)", r"$W' \rightarrow B't $" "\n(3+3)", r"$W_{kk}' \rightarrow RW $" "\n(4+2)",
        r"$Z' \rightarrow T'T'$" "\n(5+5)", r"$Y \rightarrow HH$" "\n(6+6)"  ])
    hep.cms.label( data = True, lumi = 138)


    exp = plt.errorbar(-999, -999, yerr = 1, fmt = 'x', color = 'gray', label = label, capsize = 2.0) 
    obs = plt.plot(-999, -999, linewidth = 0, marker = 's', color = 'gray')[0]
    legend_elems2 = [exp, obs]
    labels2 = ['Expected', 'Observed']

    leg1 = plt.legend(legend_elems, labels,  ncol = 2, loc = 'upper left', bbox_to_anchor = (0.3, 1.01), numpoints = 1, handler_map={tuple: HandlerTuple(ndivide=None)})
    leg2 = plt.legend(legend_elems2, labels2,  ncol = 1, loc = 'upper left', bbox_to_anchor = (-0.02, 0.95))
    for text in leg2.get_texts(): text.set_color("gray")
    plt.gca().add_artist(leg1)
    plt.gca().add_artist(leg2)


    plt.savefig(fout , bbox_inches="tight")
    plt.close()



labels = ['Inclusive', 'CWoLa Hunting', 'TNT', 'CATHODE', 'CATHODE-b', 'VAE-QR', 'QUAK']
colors = ['black', 'blue', 'green', 'purple', 'magenta', 'red', 'orange']

n_sigs = 5
n_methods = 7

rng = np.random

rands = rng.uniform(low = 0.5, high = 1.0, size = (n_sigs, n_methods))
sig_scalings = [20., 30., 30., 40., 20.]
method_scalings = [0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
method_obs_scaling = [1.05, 1.12, 0.95, 1.03, 1.08, 0.93, 0.88]

vals = []
for j in range(n_methods):
    method_vals = []
    for i in range(n_sigs):
        y_exp = rands[i,j] * sig_scalings[i] * method_scalings[j]
        y_obs = y_exp * method_obs_scaling[j]
        y_exp_up = 1.1 * y_exp
        y_exp_down = 0.9 * y_exp
        method_vals.append([y_exp, y_exp_up, y_exp_down, y_obs])
    vals.append(method_vals)


make_limit_plot(vals, labels, colors, fout = "limit_test.png")

"""
colors = ['black', 'blue', 'green']
labels = ["Inclusive", "CWoLa Hunting", "TNT"]
fs_Wp = ["inclusive_Wp3000_params.json", "cwola_Wp3000_params.json", "TNT_Wp3000_params.json"]
fs_X = ["inclusive_X3000_params.json", "cwola_X3000_params.json", "TNT_X3000_params.json"]

fs = [fs_Wp, fs_X]
fouts = ["Wp_pvals.png", "XYYp_pvals.png"]
plt_labels = ["Wp", "XYY"]

titles = [r"$W' \rightarrow B't (B' \rightarrow bZ)$", r"$X \rightarrow YY' (Y/Y' \rightarrow qq)$"]

odir = "pvals/"

fs = [["TNT_Wp3000_params.json"]]


for l_idx,flist  in enumerate(fs):
        
    for f in flist:
        with open(odir + f, 'r') as fo:
            saved_params = json.load(fo, encoding="latin-1")
        if('TNT' in f):
            nosel_fit = saved_params['fit_nosel']
            inc_obs_lim = nosel_fit['obs_lim_events']  / lumi / saved_params.preselection_eff
            inc_exp_lim = nosel_fit['exp_lim_events']  / lumi / saved_params.preselection_eff
            inc_exp_high = nosel_fit['exp_lim_1sig_high']  / lumi / saved_params.preselection_eff
            inc_exp_low = nosel_fit['exp_lim_1sig_low']  / lumi / saved_params.preselection_eff


        inj_xsecs = saved_params['injected_xsecs']
        spbs = saved_params['spbs']
        best_spb = saved_params['best_spb']

        spbs = np.array(saved_params['spbs'])
        xsecs = np.array(saved_params['injected_xsecs'])
        pvals = np.array(saved_params['pvals'])
        filtered_i = [i for i in range(len(xsecs)) if check_xsec(xsecs[i], signal = plt_labels[l_idx]) ]
        xsec = xsecs[filtered_i]
        pvals = pvals[filtered_i][xsec.argsort()]
        xsec = xsec[xsec.argsort()]
        all_pvals.append(pvals)


    make_pval_plot(xsec, all_pvals, labels, colors, title = titles[l_idx], fout = odir + fouts[l_idx])
"""
