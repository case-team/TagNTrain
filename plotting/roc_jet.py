import sys
sys.path.append('..')
from utils.TrainingUtils import *
import h5py


fin = "../data/BB_v2_3500_images/BB_images_testset.h5"
#norm_img = "../data/jet_images_BB2500_norm.h5"
norm_img = ""
j_label = "j1_"
sig_idx = 1
roc_plot_name = "%ss%i_roc.png" %(j_label, sig_idx)
sic_plot_name = "%ss%i_sic.png" % (j_label, sig_idx)

eta_cut = -1

#m_low = 2250.
#m_high = 2750.
m_low = 3150.
m_high = 3850.

single_file = True
hadronic_only = True

plot_dir = "../plots/BB_v2_M3500/july14/"
model_dir = "../models/BB_v2_M3500/july14/"

#model type: 0 is CNN, 1 is autoencoder, 2 is dense network, 5 is VAE

#f_models = ["j1_autoencoder_s1.h5",  "j1_vae_s1/",  "j1_vae_all_s1/"] 
#f_models = ["j2_autoencoder_m2500.h5",  "j2_vae_m2500/",  "j2_vae_all/"] 
f_models = [
f'{j_label}cwola_hunting_dense_deta_sig025_seed1_s{sig_idx}.h5',
f'{j_label}cwola_hunting_dense_deta_sig025_seed2_s{sig_idx}.h5',
f'{j_label}cwola_hunting_dense_deta_sig025_seed3_s{sig_idx}.h5',
f'{j_label}cwola_hunting_dense_deta_sig025_seed4_s{sig_idx}.h5',
f'{j_label}cwola_hunting_dense_deta_sig025_seed5_s{sig_idx}.h5',
        ] 
model_type = [2,2,2,2,2] 
labels = [
"CWoLa #1",  
"CWoLa #2",  
"CWoLa #3",  
"CWoLa #4",  
"CWoLa #5",  
        ]


colors = ["g", "b", "r", "gray", "purple", "pink", "orange", "m", "skyblue", "yellow", "lightcoral", "gold","olive"]
hist_labels = ["Background", "Signal"]
hist_colors = ["b", "r"]

keys = ['mjj', 'j1_images', 'j2_images']
keys = ["j1_images", "j2_images", "j1_features", "j2_features"]
x_label = j_label + "images"

if(single_file):
    num_data = -1
    data_start = 0
    data = DataReader(fin, signal_idx = sig_idx, start = data_start, stop = data_start + num_data, keys = keys, m_low = m_low, m_high = m_high, hadronic_only = hadronic_only, 
            eta_cut = eta_cut)
    data.read()
    images = data[x_label]
    Y = data['label'].reshape(-1)

else:
    bkg_start = 1000000
    n_bkg = 1000000
    sig_start = 20000
    sig_stop = -1
    if(sig_idx == 1):
        sig_stop = 30000
    #sig_stop = 25000
    d_bkg = DataReader(f_bkg, keys = keys, signal_idx = -1, start = bkg_start, stop = bkg_start + n_bkg, m_low = m_low, m_high = m_high, hadronic_only = hadronic_only, eta_cut = eta_cut )
    d_bkg.read()
    d_sig = DataReader(f_sig, keys = keys, signal_idx = -1, start = sig_start, stop = sig_stop, m_low = m_low, m_high = m_high, hadronic_only = hadronic_only, eta_cut = eta_cut )
    d_sig.read()

    im_bkg = d_bkg[x_label]
    im_sig = d_sig[x_label]
    images = np.concatenate((im_bkg, im_sig), axis = 0)
    Y = np.concatenate((np.zeros(im_bkg.shape[0], dtype=np.int8), np.ones(im_sig.shape[0], dtype=np.int8)))

bkg_events = Y < 0.1
sig_events = Y > 0.9

images = data[j_label + "images"]
dense_inputs = data[j_label + "features"]


model_scores = []
for idx,f in enumerate(f_models):

    scores = get_single_jet_scores(model_dir + f_models[idx], model_type[idx], j_images = images, j_dense_inputs = dense_inputs, batch_size = 1000)
    #hist_scores = [scores[bkg_events], scores[sig_events]]
    #make_histogram(hist_scores, hist_labels, hist_colors, 'Labeler Score', "", 100,
            #normalize = True, save = True, fname = plot_dir + f.replace('.h5', '').replace("/", "") +"_scores.png")
    model_scores.append(scores)




make_roc_curve(model_scores, Y, labels = labels, colors = colors,  logy = True, fname=plot_dir + roc_plot_name)
ymax = 8.
make_sic_curve(model_scores, Y, labels = labels, colors = colors, ymax = 10.,  fname=plot_dir + sic_plot_name)
del data
