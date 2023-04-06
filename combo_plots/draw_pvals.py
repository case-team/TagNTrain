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
        xdash = [0, xsecs[-1]]
        if(i >= 5): xdash = [xsecs[-1]/2, xsecs[-1]]
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
    plt.xlim(0, xmax * 1.08)
    hep.cms.text(" Preliminary")
    plt.legend(loc = 'lower left', title = title)
    plt.savefig(fout , bbox_inches="tight")
    plt.close()

def inside(x, xlist):
    thresh = 0.03
    for x_ref in xlist:
        diff = abs(x_ref - x)/x_ref
        if( diff < thresh): return True
    return False



def check_xsec(xsec, signal = "Wp"):
    lumi = 26.81
    if(signal == "Wp"):
        n_evts_inj = np.array([1300., 1100., 900., 700., 500., 300.])
        presel_eff = 0.498
    elif(signal == "XYY"):
        n_evts_inj = np.array([480., 360., 240., 160., 100.])
        presel_eff = 0.748


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


labels = ['Inclusive', 'CWoLa Hunting', 'TNT', 'CATHODE', 'VAE-QR', 'QUAK']
colors = ['black', 'blue', 'green', 'purple', 'red', 'orange']

fs_Wp = ["inclusive_Wp3000_params.json", "cwola_Wp3000_params.json", "TNT_Wp3000_params.json", [], [], [] ]
fs_X = ["inclusive_X3000_params.json", "cwola_X3000_params.json", "TNT_X3000_params.json", [], [], [] ] 

#CATHODE vals
fs_X[3]  = [0.424320, 0.008046, 2.43E-10, 3.96E-35, 7.34E-73]
fs_Wp[3] = [0.5, 0.5, 0.0036042878375178583, 0.00018044933946255673, 4.624866342321049e-05, 4.6335211965816587e-07]

#VAE-QR vals
fs_X[4] = [0.3, 0.3, 0.3, 0.3, 0.3] 
fs_Wp[4] = [0.3, 0.3, 0.3, 0.3, 0.3] 

#QUAK vals
fs_X[5]  = [0.005475946508677022, 3.278519482915818e-05, 5.9118509545289e-09, 8.952804533183322e-13, 2.50492289311613e-20]
fs_Wp[5] = [0.0009054009349003622, 6.5419615368228896e-06, 6.19083971768793e-11, 3.2752568900693405e-15, 8.796212141735043e-19, 4.0716871922065627e-14]

fs = [fs_Wp, fs_X]
fouts = ["Wp_pvals.png", "XYYp_pvals.png"]


for l_idx,flist  in enumerate(fs):
    all_pvals = []
    for o in flist:

        if(type(o) == str):
            with open(odir + o, 'r') as fo:
                saved_params = json.load(fo, encoding="latin-1")

            print(o)
            spbs = np.array(saved_params['spbs'])
            xsecs = np.array(saved_params['injected_xsecs'])
            print('og', xsecs)
            pvals = np.array(saved_params['pvals'])
            print('og-p', pvals)
            ilist = []
            xsecs_filt = []
            for i in range(len(xsecs)):
                if (check_xsec(xsecs[i], signal = plt_labels[l_idx]) and not inside(xsecs[i], xsecs_filt)):
                    ilist.append(i)
                    xsecs_filt.append(xsecs[i])
            xsecs_filt = np.array(xsecs_filt)
            print(xsecs_filt)
            print(xsecs_filt.argsort())
            pvals_out = pvals[ilist][xsecs_filt.argsort()]
            xsecs_out = xsecs_filt[xsecs_filt.argsort()]
            print('pval-o', pvals_out)
            print('xsec-o', xsecs_out)
            all_pvals.append(pvals_out)
        else:
            #just store list of pvals for other methods
            pvals = o
            all_pvals.append(pvals)


    print(xsecs_out)
    make_pval_plot(xsecs_out, all_pvals, labels, colors, title = titles[l_idx], fout = odir + fouts[l_idx])
