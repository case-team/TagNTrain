import sys
sys.path.append('..')
from utils.TrainingUtils import *
import fastjet as fj
import energyflow as ef
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential, load_model
from energyflow.utils import data_split, pixelate, standardize, to_categorical, zero_center
import h5py

def pj_from_PtEtaPhiM(pt, eta, phi, m):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = np.sqrt(px**2 + py**2 + pz**2 + m**2)
    return fj.PseudoJet(px, py, pz,E)



fin = "../data/jet_images.h5"
plot_dir = "../plots/mass_sculpting/"
model_dir = "../models/"

model_name = "TNT2_CNN_s10p_1p.h5"
plot_label = "TNT2_1p_"

#model types: 0 CNN (one jet), 1 auto encoder, 2 dense (one jet), 3 CNN (both jets), 4 dense (both jets)
model_type = 0
use_dense = model_type == 2 or model_type == 4
use_both = model_type == 3 or model_type == 4
new_dense = False

num_data = 100000
data_start = 1000000

js_threshholds = [90., 80., 70., 50., 0.0]
jj_threshholds = [99., 95., 90., 75., 0.0]

reduce_signal = True
signal_fraction = 0.01


if(not use_dense):
    hf_in = h5py.File(fin, "r")
    j1_images_raw = hf_in['j1_images'][data_start:data_start + num_data]
    j2_images_raw = hf_in['j2_images'][data_start:data_start + num_data]
    jet_infos = hf_in['jet_infos'][data_start:data_start + num_data]
    Y = jet_infos[:,0] #is signal bit is first bit of info
    mjj = jet_infos[:,9]

else:
    if(new_dense):
        hf_dense = h5py.File(fin, "r")
        dense_events = hf_dense['data']
        mjj = dense_events[dense_start:dense_start + num_data, 1]
        #clean_events_v2 doesn't return issig or mjj columns, so shift idxs by 2
        dense_vals = clean_events_v2(dense_events)
        idx1_start = 0
        idx1_end = 6
        idx2_end = 12
        dense_start = data_start
        print("Dense start is %i " % dense_start)
        if(dense_start < 0):
            print("Data start is %i, dense start event is %i, error, exiting \n" %data_start, dense_start_evt)
        j1_dense_inputs = dense_vals[dense_start:dense_start + num_data, idx1_start:idx1_end]
        j2_dense_inputs = dense_vals[dense_start:dense_start + num_data, idx1_end:idx2_end]


    else:
        pd_events = pd.read_hdf(fin)
        pd_events = clean_events(pd_events)
        idx1_start = 2
        idx1_end = 8
        idx2_start = 8
        idx2_end = 14
        j1_dense_inputs = pd_events.iloc[data_start:data_start + num_data, idx1_start:idx1_end].values
        j2_dense_inputs = pd_events.iloc[data_start:data_start + num_data, idx2_start:idx2_end].values

        Y = pd_events.iloc[data_start:data_start + num_data, [0]].values.reshape(-1)
        mjj = pd_events.iloc[data_start:data_start + num_data, [1]].values.reshape(-1)


if(reduce_signal):
    mask = get_signal_mask(Y, signal_fraction)
    if(use_dense):
        j1_dense_inputs = j1_dense_inputs[mask]
        j2_dense_inputs = j2_dense_inputs[mask]
    else:
        j1_images_raw = j1_images_raw[mask]
        j2_images_raw = j2_images_raw[mask]
        j1_images = np.expand_dims(j1_images_raw, axis=-1)
        j1_images = standardize(*zero_center(j1_images, np.zeros_like(j1_images)))[0]
        j2_images = np.expand_dims(j2_images_raw, axis=-1)
        j2_images = standardize(*zero_center(j2_images, np.zeros_like(j2_images)))[0]
    mjj = mjj[mask]
    Y = Y[mask]
    #j1_4vec = j1_4vec[mask]
    #j2_4vec = j2_4vec[mask]


batch_size = 1000


if(model_type <= 2):
    threshholds = js_threshholds
    j1_model = load_model(model_dir + "j1_" + model_name)
    j2_model = load_model(model_dir + "j2_" + model_name)
    if(model_type == 0):
        j1_scores = j1_model.predict(j1_images, batch_size = batch_size)
        j2_scores = j2_model.predict(j2_images, batch_size = batch_size)
    elif(model_type ==1):
        j1_reco_images = j1_model.predict(j1_images, batch_size = batch_size)
        j2_reco_images = j2_model.predict(j2_images, batch_size = batch_size)
        j1_scores =  np.mean(keras.losses.mean_squared_error(j1_reco_images, j1_images), axis=(1,2)).reshape(-1)
        j2_scores =  np.mean(keras.losses.mean_squared_error(j2_reco_images, j2_images), axis=(1,2)).reshape(-1)
    elif(model_type == 2):
        j1_scores = j1_model.predict(j1_dense_inputs, batch_size = batch_size)
        j2_scores = j2_model.predict(j2_dense_inputs, batch_size = batch_size)

    j1_scores = j1_scores.reshape(-1)
    j2_scores = j2_scores.reshape(-1)
    j1_cut_vals = [np.percentile(j1_scores, thresh) for thresh in threshholds]
    j2_cut_vals = [np.percentile(j2_scores, thresh) for thresh in threshholds]
else:
    threshholds = jj_threshholds
    if(model_type == 3): #CNN
        jj_model = load_model(model_dir + "jj_" + model_name)
        X = np.stack((j1_images_raw,j2_images_raw), axis = -1)
        X = standardize(*zero_center(X))[0]

    if(model_type == 4): #Dense
        jj_model = load_model(model_dir + "jj_" + model_name)
        X = np.append(j1_dense_inputs, j2_dense_inputs, axis = -1)
        print(X.shape)
        scores = jj_model.predict(X, batch_size = batch_size).reshape(-1)

    jj_scores = jj_model.predict(X, batch_size = batch_size).reshape(-1)
    jj_cut_vals = [np.percentile(jj_scores, thresh) for thresh in jj_threshholds]










save_figs = True
labels = ['background', 'signal']
#colors = ['b', 'r', 'g', 'purple', 'pink', 'black', 'magenta', 'pink']
colors = ["g", "gray", "b", "r","m", "skyblue", "pink"]
colors_sculpt = []
colormap = cm.viridis
normalize = mcolors.Normalize(vmin = 0., vmax=100.)



n_m_bins = 30
n_pt_bins = 30
m_range = (1500., 7000.)
pt_range = (500., 4000.)

#j1_pt = j1_4vec[:,0]
#j2_pt = j2_4vec[:,0]


sig_events = (Y > 0.9)
bkg_events = (Y < 0.1)
mjj_dists = []
j1_pt_dists = []
j2_pt_dists = []
dist_labels = []
#mjj_dists.append(mjj[bkg_events])

for i,thresh in enumerate(threshholds):
    print("Idx %i, thresh %.2f "  %(i, thresh))
    percentile = (100. - thresh)
    if(thresh == 0.): label = "No Selection"
    else: label = "X =%.0f%%" % percentile
    dist_labels.append(label)
    if(model_type <= 2):
        pass_cut = (j1_scores > j1_cut_vals[i]) & (j2_scores > j2_cut_vals[i])
    else:
        pass_cut = (jj_scores > jj_cut_vals[i])

    colors_sculpt.append(colormap(normalize(thresh)))
    sig_pass_cut = pass_cut & sig_events
    bkg_pass_cut = pass_cut & bkg_events
    masses = [mjj[bkg_events & pass_cut], mjj[sig_events & pass_cut]]
    mjj_dists.append(mjj[bkg_events & pass_cut])
    #j1_pt_dists.append(j1_pt[bkg_pass_cut])
    #j2_pt_dists.append(j2_pt[bkg_pass_cut])
    #FIXME Turn into histogram first
    #js_div = JS_Distance(mjj_dists[0], mjj[bkg_events & pass_cut])
    #print("Mean mass, JS divergence is : ", np.mean(mjj_dists[i]), js_div)
    make_histogram(masses, labels, ['b','r'], 'Dijet Mass (GeV)', label, n_m_bins, 
            stacked = True, save = save_figs,  h_range = m_range, fname=plot_dir + plot_label + "%.0fpcut_mass.png" %percentile)

#j1_pt_dists = [j1_pt[bkg_events & (j1_scores > j1_cut_vals[i]) & (j2_scores > j2_cut_vals[i])] for i in range(len(threshholds))] 
#j2_pt_dists = [j2_pt[bkg_events & (j1_scores > j1_cut_vals[i]) & (j2_scores > j2_cut_vals[i])] for i in range(len(threshholds))] 

make_histogram(mjj_dists, dist_labels, colors_sculpt, 'Dijet Mass (GeV)', "", n_m_bins,
               normalize=True, yaxis_label ="Arbitrary Units", save = save_figs,  h_range = m_range, fname=plot_dir + plot_label + "qcd_mass_sculpt.png")

#make_histogram(j1_pt_dists, dist_labels, colors_sculpt, 'J1 Pt(GeV)', "QCD J1 Pt distribution", n_pt_bins,
#               normalize=True, save = save_figs,  h_range = pt_range, fname=plot_dir + plot_label + "qcd_j1_pt_sculpt.png")
#make_histogram(j2_pt_dists, dist_labels, colors_sculpt, 'J2 Pt(GeV)', "QCD J2 Pt distribution", n_pt_bins,
#               normalize=True, save = save_figs,  h_range = pt_range, fname=plot_dir + plot_label + "qcd_j2_pt_sculpt.png")

