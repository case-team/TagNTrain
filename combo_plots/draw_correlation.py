import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
import argparse
import sys
sys.path.append('..')
#from utils.PlotUtils import *
import mplhep as hep
import tensorflow as tf
from sklearn.preprocessing import QuantileTransformer

#from https://github.com/gkasieczka/DisCo/blob/master/Disco_tf.py
def distance_corr(var_1, var_2, normedweight = None, power=1):
    """var_1: First variable to decorrelate (eg mass)
    var_2: Second variable to decorrelate (eg classifier output)
    normedweight: Per-example weight. Sum of weights should add up to N (where N is the number of examples)
    power: Exponent used in calculating the distance correlation
    
    va1_1, var_2 and normedweight should all be 1D tf tensors with the same number of entries
    
    Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
    """

    var_1 = tf.cast(var_1, tf.float64)
    var_2 = tf.cast(var_2, tf.float64)

    if(normedweight is None): 
        normedweight = tf.ones_like(var_1)

    xx = tf.reshape(var_1, [-1, 1])
    xx = tf.tile(xx, [1, tf.size(var_1)])
    xx = tf.reshape(xx, [tf.size(var_1), tf.size(var_1)])
 
    yy = tf.transpose(xx)
    amat = tf.math.abs(xx-yy)
    
    xx = tf.reshape(var_2, [-1, 1])
    xx = tf.tile(xx, [1, tf.size(var_2)])
    xx = tf.reshape(xx, [tf.size(var_2), tf.size(var_2)])
    
    yy = tf.transpose(xx)
    bmat = tf.math.abs(xx-yy)

   
    amatavg = tf.reduce_mean(amat*normedweight, axis=1)
    bmatavg = tf.reduce_mean(bmat*normedweight, axis=1)
 
    minuend_1 = tf.tile(amatavg, [tf.size(var_1)])
    minuend_1 = tf.reshape(minuend_1, [tf.size(var_1), tf.size(var_1)])
    minuend_2 = tf.transpose(minuend_1)
    Amat = amat-minuend_1-minuend_2+tf.reduce_mean(amatavg*normedweight)

    minuend_1 = tf.tile(bmatavg, [tf.size(var_2)])
    minuend_1 = tf.reshape(minuend_1, [tf.size(var_2), tf.size(var_2)])
    minuend_2 = tf.transpose(minuend_1)
    Bmat = bmat-minuend_1-minuend_2+tf.reduce_mean(bmatavg*normedweight)

    ABavg = tf.reduce_mean(Amat*Bmat*normedweight,axis=1)
    AAavg = tf.reduce_mean(Amat*Amat*normedweight,axis=1)
    BBavg = tf.reduce_mean(Bmat*Bmat*normedweight,axis=1)
   
    if power==1:
        dCorr = tf.reduce_mean(ABavg*normedweight)/tf.math.sqrt(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight))
    elif power==2:
        dCorr = (tf.reduce_mean(ABavg*normedweight))**2/(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight))
    else:
        dCorr = (tf.reduce_mean(ABavg*normedweight)/tf.math.sqrt(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight)))**power
  
    with tf.Session() as sess:
        out = dCorr.eval()
    return out



def make_corr_plot(x, y, color, axis_names, title, fname= "", y_range = (0., 1.2), x_range = (0., 1.2)  ):

    fig, ax = plt.subplots()
    alpha = 0.5
    size = 10.0
    ax.scatter(x,y, alpha = alpha, c = color, s=size)

    lin_corr = np.corrcoef(x,y)[0,1]
    dist_corr = distance_corr(x,y)
    print(lin_corr, dist_corr)
    text_str = r'$\rho $ = %.3f, Disco %.3f' % (lin_corr, dist_corr)
    plt.title(title, fontsize = 22)
    plt.annotate(text_str, xy = (0.25, 0.92),  xycoords = 'axes fraction', fontsize=14,
            bbox = dict(boxstyle='round', fc = 'w'))

    ax.set_xlabel(axis_names[0], fontsize=14)
    ax.set_ylabel(axis_names[1], fontsize=14)
    plt.tick_params(axis='y', labelsize=12)
    plt.tick_params(axis='x', labelsize=12)
    plt.ylim(y_range)
    plt.xlim(x_range)
    if(fname != ""):
        print("saving %s" % fname)
        plt.savefig(fname)


#restrict to range 0 to 1
def normalize(x, gaus = False):
    x_max = np.amax(x)
    x_min = np.amin(x)

    xout = (x - x_min)/(x_max - x_min)
    if(gaus):
        qt = QuantileTransformer(copy = True, output_distribution = 'normal')
        xout = qt.fit_transform(xout.reshape(-1,1)).reshape(-1)

    return xout 


def make_comparison(f1_name, f2_name, label1, label2, title, fout):
    if(".h5" in f1_name): f1 = h5py.File(f1_name)
    else: f1 = np.load(f1_name)

    if(".h5" in f2_name): f2 = h5py.File(f2_name)
    else: f2 = np.load(f2_name)

    scores1 = f1['scores'][:].reshape(-1)
    scores2 = f2['scores'][:].reshape(-1)

    #should be same
    mjj1  = f1['mjj'][:].reshape(-1)
    mjj2  = f2['mjj'][:].reshape(-1)

    #correct for TeV to GeV
    if(np.mean(mjj1) < 100.) : mjj1 *= 1000.
    if(np.mean(mjj2) < 100.) : mjj2 *= 1000.

    diff = np.mean(mjj1 - mjj2)

    if(scores1.shape[0] != scores2.shape[0] or (diff > 0.001)):
        print("Non matching scores for %s %s" % (f1_name, f2_name))
        print(diff)

    gaus = True
    scores1 = normalize(scores1, gaus)
    scores2 = normalize(scores2, gaus)

    ax1_label = "%s Score" % label1
    ax2_label = "%s Score" % label2

    if(gaus):
        y_range = x_range = (-3., 3.)
    else:
        x_range = (0., 1.)
        y_range(0., 1.2)

    make_corr_plot(scores1, scores2, 'blue', [ax1_label, ax2_label], title, fout, x_range = x_range, y_range = y_range)
    


d = "correlation/"
TNT_files = ["TNT_XYY_scores.npz", "TNT_Wp_scores.npz", "TNT_bkg_scores.npz"]
cwola_files = ["cwola_XYY_scores.npz", "cwola_Wp_scores.npz", "cwola_bkg_scores.npz"]
cathode_files = ["cathode_scores_x.npy", "cathode_scores_wprime.npy", "cathode_scores_bg_only.npy"]
quak_files = ["QUAK_XYY.npz", "QUAK_Wp.npz", "QUAK_bkg.npz"]
vae_files =  ["VAE_XYY.h5", "VAE_Wp.h5", "VAE_bkg.h5"]

d_out = "correlation/gaus/"


labels = ['XToYY', 'WpToBpt', 'QCD']


#for i1 in range(4):
#    for i2 in range(i1):
#        if(i1 == i2): continue
#        title = "%i %i CMP" % (i1, i2)
#        label = "Wp"
#        make_comparison(d + "TNT_k%i_Wp_scores.npz" % i1, d + "TNT_k%i_Wp_scores.npz" %i2, "TNT" + str(i1), "TNT" + str(i2), title, d + "%s_TNT%i_TNT%i_corr.png" % (label, i1, i2))

for i,label  in enumerate(labels):
    print(i)
    #if(i==0): continue
    #if(i==1): continue
    title = label + " Score Correlation"
    make_comparison(d + TNT_files[i], d + cwola_files[i], "TNT", "CWoLa Hunting", title, d_out + "%s_TNT_cwola_corr.png" % (label))
    make_comparison(d + TNT_files[i], d + cathode_files[i], "TNT", "CATHODE", title, d_out + "%s_TNT_cathode_corr.png" % (label))
    make_comparison(d + TNT_files[i], d + quak_files[i], "TNT", "QUAK", title, d_out + "%s_TNT_quak_corr.png" % (label))
    make_comparison(d + TNT_files[i], d + vae_files[i], "TNT", "VAE", title, d_out + "%s_TNT_vae_corr.png" % (label))

    make_comparison(d + cwola_files[i], d + cathode_files[i], "CWoLa Hunting", "CATHODE", title, d_out + "%s_cwola_cathode_corr.png" % (label))
    make_comparison(d + cwola_files[i], d + quak_files[i], "CWoLa Hunting", "QUAK", title, d_out + "%s_cwola_quak_corr.png" % (label))
    make_comparison(d + cwola_files[i], d + vae_files[i], "CWoLa Hunting", "VAE", title, d_out + "%s_cwola_vae_corr.png" % (label))


    make_comparison(d + quak_files[i], d + cathode_files[i], "QUAK", "CATHODE", title, d_out + "%s_quak_cathode_corr.png" % (label))
    make_comparison(d + quak_files[i], d + vae_files[i], "QUAK", "VAE", title, d_out + "%s_quak_vae_corr.png" % (label))

    make_comparison(d + vae_files[i], d + cathode_files[i], "VAE", "CATHODE", title, d_out + "%s_vae_cathode_corr.png" % (label))

