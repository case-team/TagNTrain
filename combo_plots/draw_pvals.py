import h5py
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.special import erf
import argparse
import sys
sys.path.append('..')
from utils.TrainingUtils import *
import mplhep as hep


def make_pval_plot(xsecs, pval_lists, labels, colors, title = "", fout = ""):
    #pval plot
    plt.style.use(hep.style.CMS)
    plt.figure(figsize=(12,9))

    overflow_thresh =  3e-13
    #Draw p-vals
    for i in range(len(pval_lists)):
        overflow_xs = []
        overflow_vals = []
        regular_xs = []
        regular_vals = []
        #special marker for overflows
        for j in range(len(pval_lists[i])):
            if( pval_lists[i][j] < overflow_thresh):
                overflow_xs.append(xsecs[j])
                overflow_vals.append(overflow_thresh * 2)
            else:
                regular_xs.append(xsecs[j])
                regular_vals.append(pval_lists[i][j])

        #print("reg", regular_xs, regular_vals)
        #print("over", overflow_xs, overflow_vals)
        plt.plot(regular_xs, regular_vals,  '-o',  markersize = 10.0, c = colors[i], label = labels[i],)
        plt.plot(overflow_xs, overflow_vals, '-v',  marker = "v", markersize = 15.0, c= colors[i])
        if(len(overflow_xs) > 0):
            #line connecting two sets of points
            plt.plot([regular_xs[-1], overflow_xs[0]], [regular_vals[-1], overflow_vals[0]], "-", color = colors[i])

    #Draw lines for different sigma
    corr_sigma = []
    nsig = 7
    for i in range(1, nsig+1):
        tmp = 0.5-(0.5*(1+erf(i/np.sqrt(2)))-0.5*(1+erf(0/np.sqrt(2))))
        #if i == 1:
            #print(f"This should be 0.1586553: {tmp}")
        corr_sigma.append(tmp)

    xmax = np.amax(xsecs)
    for i in range(len(corr_sigma)):
        xdash = [xsecs[0]*0.5, xsecs[-1]]
        #if(i >= 4): xdash = [xsecs[-1]*0.4, xsecs[-1]]
        y = np.full_like(xdash, corr_sigma[i])
        plt.plot(xdash, y, color="red", linestyle="dashed", linewidth=0.7)
        plt.text(xmax*1.02, corr_sigma[i]*1.25, r"{} $\sigma$".format(i+1), fontsize=14, color="red")

    #plt.text(0.05, 0.9, title, fontsize = 18, transform = plt.gca().transAxes)
    plt.xlabel(r"Cross Section (fb)")
    plt.ylabel("p-value")
    plt.yscale("log")
    plt.ylim(overflow_thresh, 1)
    ax = plt.gca()
    #ax.set_yticks([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1])
    y_minor = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0)*0.1, numticks=10)
    ax.yaxis.set_minor_locator(y_minor)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    plt.xlim(-0.35*xmax, xmax * 1.08)
    #hep.cms.text(" Preliminary")
    hep.cms.label( data = True, lumi = 138)
    plt.legend(loc = 'center left', title = title, fontsize = 18)
    plt.savefig(fout , bbox_inches="tight")
    plt.close()

def inside(x, xlist):
    thresh = 0.03
    for x_ref in xlist:
        diff = abs(x_ref - x)/x_ref
        if( diff < thresh): return True
    return False


lumi = 26.81
Wp_evts_inj = np.array([300., 500., 700., 900., 1100., 1300.])
Wp_presel_eff = 0.498

XYY_evts_inj = np.array([100., 160., 240., 360., 480.])
XYY_presel_eff = 0.748

def check_xsec(xsec, signal = "Wp"):
    if(signal == "Wp"):
        n_evts_inj = Wp_evts_inj
        presel_eff = Wp_presel_eff
    elif(signal == "XYY"):
        n_evts_inj = XYY_evts_inj
        presel_eff = XYY_presel_eff


    xsecs_inj = n_evts_inj / lumi  / presel_eff

    thresh = 0.001
    for xsec_i in xsecs_inj:
        diff = abs(xsec - xsec_i)/xsec
        if( diff < thresh): 
            return True

        
    return False



plt_labels = ["Wp", "XYY"]
titles = [r"$W' \rightarrow B't (B' \rightarrow bZ)$", r"$X \rightarrow YY' (Y/Y' \rightarrow qq)$"]
odir = "pvals/"


labels = ['Inclusive', 'CWoLa Hunting', 'TNT', 'CATHODE', 'CATHODE-b', 'VAE-QR', 'QUAK: General', 'QUAK: Model Specific']
colors = ['black', 'blue', 'green', 'purple', 'magenta', 'red', 'orange', 'gray']

#fs_Wp = [[], "cwola_Wp3000_params.json", "TNT_Wp3000_params.json", [], [], [], [] ]
#fs_X = [[], "cwola_X3000_params.json", "TNT_X3000_params.json", [], [], [], [] ] 

fs_Wp = [[]]*8
fs_X = [[]]*8


#inclusive 
fs_X[0] = [0.5, 0.5, 0.5, 0.5, 0.0655]
fs_Wp[0] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

#cwola
fs_X[1] =  [5.00000000e-01, 2.84041990e-01, 3.42473959e-01, 4.02455957e-01, 2.80399922e-04]
fs_Wp[1] = [5.00000000e-01, 5.00000000e-01, 1.21356794e-01, 3.52915423e-02, 9.11181820e-06, 1.9426e-09] 

#TNT
fs_X[2] = [1.43251192e-01, 2.63162995e-02, 1.66270864e-06, 1.38861910e-15, 2.88545364e-18,]
fs_Wp[2] = [3.45744540e-01, 2.32689374e-02, 7.34220798e-05, 3.99141524e-08, 1.13942470e-10, 9.69697101e-16]

#CATHODE vals
fs_X[3]  = [0.227733785468067, 1.3553238704222537e-05, 1.3350653915722432e-11, 0, 0]
fs_Wp[3] = [0.38774954658769256, 0.09673829685792323, 0.02346414272060615, 0.01263124821294892, 0.0003484680650076566, 1.8194007541216806e-05]


#CATHODE-b vals
fs_X[4] = [ 0.5, 0.024466944412907338, 1.1282044729167672e-05, 9.768557109794338e-32, 5.770603951398898e-62]
fs_Wp[4] = [0.2504952562764966 , 0.01799966067621161 , 0.0024750722732585157 , 7.664756152381592e-07 , 2.75510044142784e-10 , 2.395397106456989e-19]

#VAE-QR vals
fs_X[5] = [0.5, 0.470, 0.220, 0.0388, 0.0053] 
fs_Wp[5] = [0.21, 0.099, 0.095, 0.0135, 0.0151, 0.000279] 

#QUAK vals
fs_X[6]  = [0.00017847156614762385, 4.627340705203891e-07, 9.183889271214063e-10, 1.3519075736748065e-21, 3.726752847109815e-24]
fs_Wp[6] = [0.014529040565424547, 0.007918894920774956, 5.5576386611810706e-05, 3.689073304061652e-08, 3.1774383879140555e-08, 1.0004245087099021e-07]


#QUAK model specific
fs_X[7]  = [9.84e-7, 1.2e-13, 2.22e-24, 2.54e-34, 1.17e-51]
fs_Wp[7] = [1.82e-5, 7.78e-11, 2.23e-24, 7.1e-24, 4.53e-44, 4.5e-24]

#supervised + DISCO
#fs_X[8]  = [5e-05, 4.75e-08, 7.34e-18, 7.8e-40, 3.8e-45]
#fs_Wp[8] = [0.024, 2.98e-6, 2.37e-9, 9.836e-11, 1.48e-11, 7.07e-17]

fs = [fs_Wp, fs_X]
fouts = ["Wp_pvals.png", "XYYp_pvals.png"]


for l_idx,flist  in enumerate(fs):
    all_pvals = []

    if(l_idx == 0): xsecs_inj = Wp_evts_inj / lumi  / Wp_presel_eff
    else: xsecs_inj = XYY_evts_inj / lumi  / XYY_presel_eff


    for o in flist:

        if(type(o) == str):
            with open(odir + o, 'r') as fo:
                saved_params = json.load(fo, encoding="latin-1")

            spbs = np.array(saved_params['spbs'])
            xsecs = np.array(saved_params['injected_xsecs'])
            pvals = np.array(saved_params['pvals'])
            ilist = []
            xsecs_filt = []
            for i in range(len(xsecs)):
                if (check_xsec(xsecs[i], signal = plt_labels[l_idx]) and not inside(xsecs[i], xsecs_filt)):
                    ilist.append(i)
                    xsecs_filt.append(xsecs[i])
            xsecs_filt = np.array(xsecs_filt)
            pvals_out = pvals[ilist][xsecs_filt.argsort()]
            xsecs_out = xsecs_filt[xsecs_filt.argsort()]
            all_pvals.append(pvals_out)
            print(o, pvals_out)
        else:
            #just store list of pvals for other methods
            pvals = o
            all_pvals.append(pvals)


    print(xsecs_inj)
    make_pval_plot(xsecs_inj, all_pvals, labels, colors, title = titles[l_idx], fout = odir + fouts[l_idx])
