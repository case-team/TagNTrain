import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
import argparse
import sys
sys.path.append('..')
from utils.PlotUtils import *
import mplhep as hep
import tensorflow as tf

from sklearn.preprocessing import QuantileTransformer

def transform(x):
    qt = QuantileTransformer(copy = True, output_distribution = 'uniform')
    xout = qt.fit_transform(x.reshape(-1,1)).reshape(-1)
    return xout


def get_scores(f):
    if(".h5" in f): f1 = h5py.File(f)
    else: f1 = np.load(f)

    scores1 = f1['scores'][:].reshape(-1)
    return scores1

colors = ["g", "b", "r", "gray", "purple", "pink", "orange", "m", "skyblue", "yellow"]

#colors = ['black', '#F739F2', '#702963', '#228B22', '#0271BB', '#E2001A', '#FCB40C', '#949494', 'sienna','tan', 'cyan', 'deepskyblue' ]
labels = ["TNT", "CWoLa Hunting", "CATHODE", "QUAK", "VAE", "Combined: Multiplication", "Combined: Addition", "Combined: Max"]

XYY_score_files = ["TNT_XYY_scores.npz", "cwola_XYY_scores.npz", "cathode_scores_x.npy", "QUAK_XYY.npz", "VAE_XYY.h5"]
Wp_score_files = ["TNT_Wp_scores.npz", "cwola_Wp_scores.npz", "cathode_scores_wprime.npy", "QUAK_Wp.npz", "VAE_Wp.h5"]
bkg_score_files = ["TNT_bkg_scores.npz", "cwola_bkg_scores.npz", "cathode_scores_bg_only.npy", "QUAK_bkg.npz", "VAE_bkg.h5"]


XYY_scores = [get_scores("correlation/" + f) for f in XYY_score_files]
Wp_scores = [get_scores("correlation/" + f) for f in Wp_score_files]
bkg_scores = [get_scores("correlation/" + f) for f in bkg_score_files]


print(XYY_scores[0].shape)
print(bkg_scores[0].shape)
XYY_cmp = [transform(np.concatenate((XYY_scores[i], bkg_scores[i]))) for i in range(len(XYY_scores))]
Wp_cmp = [transform(np.concatenate((Wp_scores[i], bkg_scores[i]))) for i in range(len(Wp_scores))]


XYY_scores_comb_add = np.sum(XYY_cmp, axis=0)
XYY_scores_comb_mult = np.prod(XYY_cmp, axis=0)
XYY_scores_comb_max = np.max(XYY_cmp, axis=0)

print(XYY_cmp[0].shape)
print(XYY_scores_comb_add.shape)
print(XYY_scores_comb_mult.shape)

Wp_scores_comb_add = np.sum(Wp_cmp, axis=0)
Wp_scores_comb_mult = np.prod(Wp_cmp, axis=0)
Wp_scores_comb_max = np.max(Wp_cmp, axis=0)

XYY_cmp.append(XYY_scores_comb_mult)
XYY_cmp.append(XYY_scores_comb_add)
XYY_cmp.append(XYY_scores_comb_max)

Wp_cmp.append(Wp_scores_comb_mult)
Wp_cmp.append(Wp_scores_comb_add)
Wp_cmp.append(Wp_scores_comb_max)

XYY_labels = np.concatenate((np.ones_like(XYY_scores[0]), np.zeros_like(bkg_scores[0])))
Wp_labels = np.concatenate((np.ones_like(Wp_scores[0]), np.zeros_like(bkg_scores[0])))


make_sic_curve(XYY_cmp, XYY_labels, colors=colors, labels=labels, fname="combine_anomaly/XYY_sic.png", eff_min=0.01, ymax=10)
make_sic_curve(Wp_cmp, Wp_labels, colors=colors, labels=labels, fname="combine_anomaly/Wp_sic.png", eff_min=0.01, ymax=10)
    


