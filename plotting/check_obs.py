import sys
sys.path.append('..')
from utils.TrainingUtils import *
import fastjet as fj
import energyflow as ef
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential, load_model
from energyflow.utils import data_split, pixelate, standardize, to_categorical, zero_center
import h5py
from optparse import OptionParser
from optparse import OptionGroup


def scatter_plot(pts, colors, y_label = "", x_label = "", fname = "", alpha = 0.5, size = 0.4, fontsize=16):
    fig, ax = plt.subplots()
    for i in range(len(pts)):
        ax.scatter(pts[i][0], pts[i][1], alpha = alpha, s=size, c = colors[i])
    if(y_label != ""): ax.set_ylabel(y_label, fontsize = fontsize)
    if(x_label != ""): ax.set_xlabel(x_label, fontsize = fontsize)
    plt.tick_params(axis='y', labelsize=12)
    plt.tick_params(axis='x', labelsize=12)
    if(fname != ""): plt.savefig(fname)



fin = "../data/jet_images.h5"
dense_fin = "../data/events_cwbh.h5"
plot_dir = "../plots/"
model_dir = "../models/"



#model types: 0 CNN (one jet), 1 auto encoder, 2 dense (one jet), 3 CNN (both jets), 4 dense (both jets)
model_name = "TNT2_CNN_s10p_1p.h5"
plot_header = "TNT2_CNN_corr_"
model_type = 0


compare_truth = True
mjj_window = False
mjj_low = 3300
mjj_high = 3700

num_data = 100000
data_start = 1000000
#what event the dense file starts with
dense_start_evt = 000000

#two_thresh = 97
#one_thresh = 99.5

two_thresh = 92
one_thresh = 99

new_dense = False
filt_sig = True
sig_frac = 0.01


use_dense = (model_type == 2 or model_type == 4)

if(not use_dense):
    hf_in = h5py.File(fin, "r")
    j1_images_raw = hf_in['j1_images'][data_start:data_start + num_data]
    j2_images_raw = hf_in['j2_images'][data_start:data_start + num_data]

    jet_infos = hf_in['jet_infos'][data_start:data_start + num_data]
    Y = jet_infos[:,0] #is signal bit is first bit of info
    j1_4vec = jet_infos[:,1:5]
    j2_4vec = jet_infos[:,5:9]
    mjj = jet_infos[:, -1]

    #print(mjj[:20])


if(new_dense):
    hf_dense = h5py.File(dense_fin, "r")
    dense_events = hf_dense['data']
    dense_vals_clean = clean_events_v2(dense_events)
    idx1_start = 2
    idx1_end = 8
    idx2_end = 14
    dense_start = data_start - dense_start_evt
    print("Dense start is %i " % dense_start)
    if(dense_start < 0):
        print("Data start is %i, dense start event is %i, error, exiting \n" %data_start, dense_start_evt)
    j1_dense_inputs = dense_events[dense_start:dense_start + num_data, idx1_start:idx1_end]
    j2_dense_inputs = dense_events[dense_start:dense_start + num_data, idx1_end:idx2_end]
    mjj = dense_events[dense_start:dense_start + num_data, 1]
    #print(mjj[:20])
    Y = dense_events[dense_start:dense_start + num_data, 0]

    idx1_start -=2
    idx1_end -=2
    idx2_end -=2
    j1_dense_inputs_clean = dense_vals_clean[dense_start:dense_start + num_data, idx1_start:idx1_end]
    j2_dense_inputs_clean = dense_vals_clean[dense_start:dense_start + num_data, idx1_end:idx2_end]
    jj_dense_inputs_clean =  dense_vals_clean[dense_start:dense_start + num_data, idx1_start:idx2_end]
else:
    idx1_start = 2
    idx1_end = 8
    idx2_end = 14
    pd_events = pd.read_hdf(dense_fin, key = "data")
    #pd_events = clean_events(pd_events)
    X_test = pd_events.iloc[data_start:data_start + num_data, idx1_start:idx2_end].values
    y_test = pd_events.iloc[data_start:data_start + num_data, [0]].values

    j1_dense_inputs = pd_events.iloc[data_start:data_start + num_data, idx1_start:idx1_end].values
    j2_dense_inputs = pd_events.iloc[data_start:data_start + num_data, idx1_end:idx2_end].values

#print(dense_mjj[:20])



if(filt_sig):
    mask = get_signal_mask(Y, sig_frac)
    j1_images_raw = j1_images_raw[mask]
    j2_images_raw = j2_images_raw[mask]
    j1_dense_inputs = j1_dense_inputs[mask]
    j2_dense_inputs = j2_dense_inputs[mask]
    Y = Y[mask]
    j1_4vec = j1_4vec[mask]
    j2_4vec = j2_4vec[mask]
    mjj = mjj[mask]


j1_Ms = j1_dense_inputs[:,0]
j2_Ms = j2_dense_inputs[:,0]

j1_vars = j1_dense_inputs[:,1:]
j2_vars = j1_dense_inputs[:,1:]

batch_size = 1000


if(model_type <= 2):
    j1_model = load_model(model_dir + "j1_" + model_name)
    j2_model = load_model(model_dir + "j2_" + model_name)
    if(model_type == 0):

        j1_images = np.expand_dims(j1_images_raw, axis=-1)
        j1_images = standardize(*zero_center(j1_images, np.zeros_like(j1_images)))[0]
        j2_images = np.expand_dims(j2_images_raw, axis=-1)
        j2_images = standardize(*zero_center(j2_images, np.zeros_like(j2_images)))[0]
        j1_scores = j1_model.predict(j1_images, batch_size = batch_size)
        j2_scores = j2_model.predict(j2_images, batch_size = batch_size)
    elif(model_type ==1):

        j1_images = np.expand_dims(j1_images_raw, axis=-1)
        j1_images = standardize(*zero_center(j1_images, np.zeros_like(j1_images)))[0]
        j2_images = np.expand_dims(j2_images_raw, axis=-1)
        j2_images = standardize(*zero_center(j2_images, np.zeros_like(j2_images)))[0]

        j1_reco_images = j1_model.predict(j1_images, batch_size = batch_size)
        j2_reco_images = j2_model.predict(j2_images, batch_size = batch_size)
        j1_scores =  np.mean(keras.losses.mean_squared_error(j1_reco_images, j1_images), axis=(1,2)).reshape(-1)
        j2_scores =  np.mean(keras.losses.mean_squared_error(j2_reco_images, j2_images), axis=(1,2)).reshape(-1)
    elif(model_type == 2):
        j1_scores = j1_model.predict(j1_dense_inputs_clean, batch_size = batch_size)
        j2_scores = j2_model.predict(j2_dense_inputs_clean, batch_size = batch_size)

    j1_scores = j1_scores.reshape(-1)
    j2_scores = j2_scores.reshape(-1)
    j1_cut1 = np.percentile(j1_scores, one_thresh)
    j2_cut1 = np.percentile(j2_scores, one_thresh)
    j1_cut2 = np.percentile(j1_scores, two_thresh)
    j2_cut2 = np.percentile(j2_scores, two_thresh)

    j1_pass_cut = (j1_scores > j1_cut1) 
    j2_pass_cut = (j2_scores > j2_cut1)
    both_pass = (j1_scores > j1_cut2) & (j2_scores > j2_cut2)
else:
    jj_model = load_model(model_dir + "jj_" + model_name)
    jj_model = load_model(model_dir + "jj_" + model_name)
    if(model_type == 3):
        X = np.stack((j1_images_raw,j2_images_raw), axis = -1)
        X = standardize(*zero_center(X))[0]
        jj_scores = jj_model.predict(X, batch_size = batch_size).reshape(-1)
    else:
        jj_scores = jj_model.predict(jj_dense_inputs_clean, batch_size = batch_size).reshape(-1)

    jj_cut = np.percentile(jj_scores, one_thresh)
    both_pass = j2_pass_cut = j1_pass_cut = (jj_scores > jj_cut)

if(mjj_window):
    pass_mjj = (mjj > mjj_low) & (mjj < mjj_high)
    j1_pass_cut = j1_pass_cut & pass_mjj
    j2_pass_cut = j2_pass_cut & pass_mjj
    both_pass = both_pass & pass_mjj

sig_events = (Y > 0.9)
bkg_events = (Y < 0.1)

labels = ['All Events', 'Top 1\%']
colors = ['gray','b', 'r']
alpha = 0.5
size = 0.4

obs_labels = ['sqrt(tau^2_1)/tau^1_1', r'$\tau_{21}$', r'$\tau_{32}$', r'$\tau_{43}$', r'$N_{trk}$']
plot_labels = ['tau_ratio.png', 'tau21.png', 'tau32.png', 'tau43.png', 'ntrk.png']

j1_Ms_cut = j1_Ms[j1_pass_cut]
j2_Ms_cut = j2_Ms[j2_pass_cut]

j1_Ms_bothcut = j1_Ms[both_pass]
j2_Ms_bothcut = j2_Ms[both_pass]

j1_Ms_true = j1_Ms[sig_events]
j2_Ms_true = j2_Ms[sig_events]

print(j1_Ms.shape, j1_Ms_cut.shape, j1_Ms_bothcut.shape, j1_Ms_true.shape)


for i in range(j1_vars.shape[1]):
    size = 0.4
    alpha = 0.5
    if(mjj_window):
        size = 1.0
        alpha = 0.8
    j1_cut_vars = (j1_vars[j1_pass_cut])[:,i]
    j2_cut_vars = (j2_vars[j2_pass_cut])[:,i]


    j1_cut2_vars = (j1_vars[both_pass])[:,i]
    j2_cut2_vars = (j2_vars[both_pass])[:,i]

    j1_true_vars = (j1_vars[sig_events])[:,i]
    j2_true_vars = (j2_vars[sig_events])[:,i]

    scatter_plot([(j1_Ms, j1_vars[:,i]), (j1_Ms_cut, j1_cut_vars)], colors = ['gray', 'lime'], size=size, alpha=alpha,
            y_label = obs_labels[i], x_label = "Jet Mass (GeV)", fname = plot_dir + "j1_" +  plot_header + plot_labels[i])

    scatter_plot([(j1_Ms, j1_vars[:,i]), (j1_Ms_bothcut, j1_cut2_vars)], colors = ['gray', 'r'],  size=size, alpha=alpha,
            y_label = obs_labels[i], x_label = "Jet Mass (GeV)", fname = plot_dir + "j1_" +  plot_header + "both_" + plot_labels[i])
    
    if(compare_truth):
        scatter_plot([(j1_Ms, j1_vars[:,i]), (j1_Ms_true, j1_true_vars)], colors = ['gray', 'b'], 
            y_label = obs_labels[i], x_label = "Jet Mass (GeV)", fname = plot_dir + "j1_" +  plot_header + "truth_" + plot_labels[i])

    scatter_plot([(j2_Ms, j2_vars[:,i]), (j2_Ms_cut, j2_cut_vars)], colors = ['gray', 'lime'], size=size, alpha=alpha,
            y_label = obs_labels[i], x_label = "Jet Mass (GeV)", fname = plot_dir + "j2_" +  plot_header + plot_labels[i])

    scatter_plot([(j2_Ms, j2_vars[:,i]), (j2_Ms_bothcut, j2_cut2_vars)], colors = ['gray', 'r'], size=size, alpha=alpha,
            y_label = obs_labels[i], x_label = "Jet Mass (GeV)", fname = plot_dir + "j2_" +  plot_header + "both_" + plot_labels[i])
    
    if(compare_truth):
        scatter_plot([(j2_Ms, j2_vars[:,i]), (j2_Ms_true, j2_true_vars)], colors = ['gray', 'b'], 
            y_label = obs_labels[i], x_label = "Jet Mass (GeV)", fname = plot_dir + "j2_" +  plot_header + "truth_" + plot_labels[i])



