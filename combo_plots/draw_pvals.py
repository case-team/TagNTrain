import h5py
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.special import erf
import argparse
import sys
sys.path.append('..')
from utils.TrainingUtils import *
import mplhep as hep


def make_pval_plot(xsecs, pval_lists, labels, colors, title = "", markers = [], leg_order = [], exclude_idxs = [], fout = ""):
    #pval plot
    plt.style.use(hep.style.CMS)
    plt.figure(figsize=(15,10))

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
        xdash = [xsecs[0]*0.85, xmax*1.01]
        #if(i >= 4): xdash = [xsecs[-1]*0.4, xsecs[-1]]
        y = np.full_like(xdash, corr_sigma[i])
        plt.plot(xdash, y, color="black", linestyle="dashed", linewidth=0.7)
        plt.text(xmax*1.02, corr_sigma[i]*0.95, r"{} $\sigma$".format(i+1), fontsize=20, color="black")


    #Draw p-vals
    overflow_thresh =  3e-13
    filled_gap = False
    entries = []
    leg_labels = []
    for i in leg_order:
        if(i in exclude_idxs): 
            print("Excluding %s" % labels[i])
            continue
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
        entry, = plt.plot(regular_xs, regular_vals,  markersize = 15 if markers[i]=='x' else 10.0, c = colors[i], label = labels[i], marker=markers[i], linestyle = linestyles[i])
        plt.plot(overflow_xs, overflow_vals, '-v',  marker = "v", markersize = 15.0, c= colors[i], linestyle = linestyles[i])
        if(len(overflow_xs) > 0):
            #line connecting two sets of points
            plt.plot([regular_xs[-1], overflow_xs[0]], [regular_vals[-1], overflow_vals[0]], "-", color = colors[i], linestyle = linestyles[i])

        #add a gap before reference models in legend
        if(linestyles[i] == 'dashed' and not filled_gap):
            entries.append(matplotlib.lines.Line2D([],[],linestyle=''))
            leg_labels.append("")
            filled_gap = True
        entries.append(entry)
        leg_labels.append(labels[i])




    #plt.text(0.05, 0.9, title, fontsize = 18, transform = plt.gca().transAxes)
    plt.xlabel(r"Cross Section (fb)", fontsize = 26)
    plt.ylabel("p-value", fontsize = 26)
    plt.yscale("log")
    plt.ylim(overflow_thresh, 1)
    #plt.vlines(xsecs[0] * 0.85, overflow_thresh, 1, colors = 'black')
    ax = plt.gca()
    #ax.set_yticks([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1])
    y_minor = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0)*0.1, numticks=10)
    ax.yaxis.set_minor_locator(y_minor)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(axis='both', which = 'major', labelsize = 24)
    plt.xlim(-0.25*xmax, xmax * 1.08)
    #hep.cms.text(" Preliminary")
    hep.cms.label( data = False)
    plt.legend(entries, leg_labels, loc = 'center left', title = title, fontsize = 22)
    plt.savefig(fout , bbox_inches="tight")
    print("Saving " + fout)
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
titles = [r"$W' \rightarrow B't  \rightarrow bZt$", r"$X \rightarrow YY' \rightarrow 4q$"]
odir = "pvals/"


labels = ['Inclusive', 'VAE-QR',  'CWoLa Hunting', 'TNT', 'CATHODE', 'CATHODE-b',  'QUAK', 
'QUAK: Model Specific', 
r'2-prong $(\tau_{21}, m_\mathrm{SD})$', r'3-prong $(\tau_{32}, m_\mathrm{SD})$']

#                   pink       purple   dark green     blue      red       orange     #gray
colors = ['black', '#F739F2', '#702963', '#228B22', '#0271BB', '#E2001A', '#FCB40C', '#949494', 'sienna','tan' ]

markers = ['o', 's', 'P', 'p', '*', 'X', 'D', 'd', 'x', 'h']
linestyles = ['solid',] * 10

linestyles[0] = 'dashed'
linestyles[7] = 'dashed'
linestyles[8] = 'dashed'
linestyles[9] = 'dashed'

#remove CATHODE-b
#excludes = [5,7] 
excludes = [] 
leg_order = [1,2,3,4,5,6,7,8,9,0]

fs_Wp = [[]]*10
fs_X = [[]]*10

#XYY spbs 1.25 2.0 3.0 4.5 6.0
#Wp spbs 3.75, 6.25, 8.75, 11.25, 13.75, 16.25



#inclusive 
fs_X[0] = [0.27316741322093174, 0.1823121286865843, 0.08858732564563399, 0.02219257708330158, 0.003797742515738056]
fs_Wp[0] = [0.328186919746016, 0.2326755760735304, 0.1625528533558429, 0.10240051056142319, 0.06256831397566243, 0.034774872241897015]


#VAE-QR vals
fs_X[1] = [0.24166, 0.1251, 0.0347, 0.008278, 0.0006178] 
fs_Wp[1] = [0.16780, 0.0362, 0.0101, 0.00127, 0.0001135, 0.0001578] 

#cwola
fs_X[2] =  [5.00000000e-01, 0.46, 0.17, 0.2, 0.00074]
fs_Wp[2] = [0.247, 0.081, 0.0025, 1.07e-5, 9.11181820e-06, 6.5e-8] 

#TNT
fs_X[3] = [0.22, 0.0344, 2.5e-6, 2.45e-14, 2.88545364e-18,]
#fs_Wp[3] = [0.116, 0.013, 5.33e-6, 1.74e-5, 6.85e-08, 4.17e-11]
fs_Wp[3] = [0.116, 0.013, 5.33e-6, 1.74e-7, 4.2e-8, 1e-11]

#CATHODE vals
fs_X[4]  = [0.227733785468067, 1.3553238704222537e-05, 1.3350653915722432e-11, 1e-20, 1e-20]
fs_Wp[4] = [0.38774954658769256, 0.19978827448576908, 0.02346414272060615, 0.01263124821294892, 0.0003484680650076566, 1.8194007541216806e-05]


#CATHODE-b vals
fs_X[5] = [ 0.2530048253986539, 0.0031843110482953074, 3.5318038160703225e-08, 1e-20, 1e-20]
fs_Wp[5] = [0.1929923442884005, 0.006321927181276887, 9.000669694225749e-05, 4.3167147165235065e-05, 1.9242211046766045e-07, 9.344568452362978e-09]


#QUAK vals
fs_X[6]  = [0.00206, 3.3e-05, 3.4e-08, 7.6e-18, 1.3e-21]
fs_Wp[6] = [0.078, 0.0072, 0.000625, 9.5e-05, 2.76e-05, 2.98e-08]


#QUAK model specific
fs_X[7]  = [2.34e-7, 4.96e-11, 5.9e-20, 2.7e-35, 6.47e-53]
fs_Wp[7] = [0.004, 9.99e-07, 1.46e-10, 8.4e-17, 3.0e-17, 2.66e-23]


#tau21 + mSD model specific
fs_X[8]  = [0.23, 0.1, 0.034, 0.00098, 5.1e-5]
fs_Wp[8] = [0.4, 0.34, 0.29, 0.24, 0.19, 0.15]

#tau32 + mSD model specific
fs_X[9]  = [0.36, 0.283, 0.195, 0.098, 0.042]
fs_Wp[9] = [0.052, 0.0208, 0.0027, 5.8e-5, 1.07e-6, 1.33e-7]

no_taus = False

#supervised + DISCO
#fs_X[8]  = [5e-05, 4.75e-08, 7.34e-18, 7.8e-40, 3.8e-45]
#fs_Wp[8] = [0.024, 2.98e-6, 2.37e-9, 9.836e-11, 1.48e-11, 7.07e-17]

fs = [fs_Wp, fs_X]
fouts = ["Wp_pvals.png", "XYYp_pvals.png"]


for l_idx,flist  in enumerate(fs):
    if(l_idx > 8 and no_taus): continue
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


    make_pval_plot(xsecs_inj, all_pvals, labels, colors, title = titles[l_idx], markers = markers, fout = odir + fouts[l_idx], exclude_idxs = excludes, leg_order = leg_order)
