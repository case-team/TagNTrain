import h5py
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
import argparse
import sys
sys.path.append('..')
from utils.TrainingUtils import *
import mplhep as hep

colors = ["g", "b", "r", "gray", "purple", "pink", "orange", "m", "skyblue", "yellow"]

do_XToYYp = False



if(do_XToYYp):

    xsec = 20
    output_name = 'XToYYp_sic_comparison.png'
    label = r"X $\to$ YY' (Y/Y' $\to$ qq)"

    f_list = [
            "sic/TNT/cwola_X3000_Y80_Yprime170_limit_june20_spb5.0_avg_tagging_effs.npz",
            "sic/TNT/TNT_X3000_Y80_Yprime170_limit_june20_spb5.0_avg_tagging_effs.npz",
            "sic/QUAK/ForOz_XYYInjection_signalTrainedXYYAndWprime_effs.npz",
            "sic/VAE/qcdSigMCOrigReco_XToYYprimeTo4Q_MX3000_MY80_MYprime170_narrowReco_effs_v3.npz",
            "sic/CATHODE/CATHODE_XYY_20fb_v2.npz" ,
            ]


else:
    xsec = 60
    output_name = 'Wprime_sic_comparison.png'
    label = r"W' $\to$ B't (B' $\to$ bZ)"
    f_list = [
            "sic/TNT/cwola_Wp3000_Bp400_limit_june20_spb10.0_avg_tagging_effs.npz",
            "sic/TNT/TNT_Wp3000_Bp400_limit_june20_spb10.0_avg_tagging_effs.npz",
            "sic/QUAK/ForOz_WprimeInjection_signalTrainedXYYAndWprime_effs.npz",
            "sic/VAE/qcdSigMCOrigReco_WpToBpT_Wp3000_Bp400_Top170_ZbtReco_effs.npz",
            #"sic/CATHODE/wprime_50fb_effs.npz" ,
            ]

labels = [
           "CWoLa Hunting (%i fb injected)" % xsec,
           "TNT (%i fb injected)" % xsec,
           "QUAK",
           "VAE",
           ]
if(do_XToYYp): labels.append( "CATHODE (%i fb injected)" % xsec)


eff_min = 1e-3

linewidth = 3



plt.style.use(hep.style.CMS)
plt.figure(figsize=fig_size)
for i in range(len(f_list)):
    f = np.load(f_list[i])
    sig_eff, bkg_eff = f['sig_eff'], f['bkg_eff']
    #sics_ = f['sics']
    #print(sics_[:10])
    mask = bkg_eff > eff_min
    bkg_eff_clip = bkg_eff[mask]
    sig_eff_clip = sig_eff[mask]
    sics = sig_eff_clip / np.sqrt(bkg_eff_clip)
    print(labels[i], np.amax(sics))
    plt.plot(bkg_eff_clip, sics, lw=linewidth, color=colors[i], label=labels[i])


#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='random chance')
fs = 24
fs_leg = 20
fs_label = 26
plt.xlim([eff_min, 1.0])
sic_max = 10.
plt.ylim([0,sic_max])
plt.xscale('log')
plt.xlabel(r'Background Efficiency ($\epsilon_{B}$)' , fontsize = fs)
plt.ylabel(r'Significance Improvement ($\frac{\epsilon_{S}}{\sqrt{\epsilon_{B}}}$)', fontsize = fs)
plt.tick_params(axis='x', labelsize=fs_leg)
plt.tick_params(axis='y', labelsize=fs_leg)
#plt.grid(axis = 'y', linestyle='--', linewidth = 0.5)
plt.text(0.002, 9, label, fontsize = fs_label)
plt.legend(loc="best", fontsize= fs_leg)
hep.cms.label( data = False)
plt.savefig(output_name)
print("Saving file to %s " % (output_name))


