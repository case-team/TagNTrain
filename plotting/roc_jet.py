import sys
sys.path.append('..')
from utils.TrainingUtils import *
import h5py


fin = "../data/BB_v2_2500_images/BB_images_testset.h5"
batch_start = -1
batch_stop = -1
sig_frac = -1
#fin = "../data/BB_v2_3500_images/"
#batch_start = 0
#batch_stop = 7
#sig_frac = 0.001
j_label = "j2"
sig_idx = 1
roc_plot_name = "%s_s%i_roc_cwola_all_kfold_qt.png" %(j_label, sig_idx)
sic_plot_name = "%s_s%i_sic_cwola_all_kfold_qt.png" % (j_label, sig_idx)

eta_cut = -1

m_low = 2250.
m_high = 2750.
#m_low = 2864.
#m_high = 4278.

#m_low = 3150.
#m_high = 3850.
#
#msig_low = 3150.
#msig_high = 3850.

single_file = True
hadronic_only = True
randsort = False

plot_dir = "../runs/cwola_40spb_fullrun/"
model_dir = "../runs/cwola_40spb_fullrun/"

#model type: 0 is CNN, 1 is autoencoder, 2 is dense network, 5 is VAE

#f_models = ["j1_autoencoder_s1.h5",  "j1_vae_s1/",  "j1_vae_all_s1/"] 
#f_models = ["j2_autoencoder_m2500.h5",  "j2_vae_m2500/",  "j2_vae_all/"] 
#f"july14/{j_label}cwola_hunting_dense_deta_sig025_seed1_s{sig_idx}.h5", f"july22/{j_label}deta_sig025_TNT0_s{sig_idx}.h5", f"july22/jrand_deta_sig025_TNT0_s{sig_idx}.h5"]
f_models = [
        #f'{j_label}cwola_ensemble_num_model1_seed1/',
        #f'{j_label}cwola_ensemble_num_model1_seed2/',
        #f'{j_label}cwola_ensemble_num_model1_seed3/',
        #f'{j_label}cwola_ensemble_num_model1_seed4/',
        #f'{j_label}cwola_ensemble_num_model1_seed5/',
        #]
#f'{j_label}cwola_hunting_dense_deta_sig025_seed1_s{sig_idx}.h5',
#f'{j_label}cwola_hunting_dense_deta_sig025_seed2_s{sig_idx}.h5',
#f'{j_label}cwola_hunting_dense_deta_sig025_seed3_s{sig_idx}.h5',
#f'{j_label}cwola_hunting_dense_deta_sig025_seed4_s{sig_idx}.h5',
#f'{j_label}cwola_hunting_dense_deta_sig025_seed5_s{sig_idx}.h5',

#f"aug18/jrand_v2_xval5_deta_sig01_TNT0_seed1_s{sig_idx}.h5",
#f"aug18/jrand_v2_xval5_deta_sig01_TNT0_seed2_s{sig_idx}.h5",
#f"aug18/jrand_v2_xval5_deta_sig01_TNT0_seed3_s{sig_idx}.h5",
#f"aug18/jrand_v2_xval5_deta_sig01_TNT0_seed4_s{sig_idx}.h5",
#f"aug18/jrand_v2_xval5_deta_sig01_TNT0_seed5_s{sig_idx}.h5",


f"{j_label}_kfold0/",
f"{j_label}_kfold1/",
f"{j_label}_kfold2/",
f"{j_label}_kfold3/",
f"{j_label}_kfold4/",


        ] 
model_type = [2,2,2,2,2,2] 
#num_ensemble = [5,5,5,5,5,5]
num_ensemble = [4,4,4,4,4]
labels = [
#"TNT #1",  
#"TNT #2",  
#"TNT #3",  
#"TNT #4",  
#"TNT #5",  

        "k-fold 0",
        "k-fold 1",
        "k-fold 2",
        "k-fold 3",
        "k-fold 4",
        ]
#use_rand_sort = [False, False, False, False, False]



colors = ["g", "b", "r", "gray", "purple", "pink", "orange", "m", "skyblue", "yellow", "lightcoral", "gold","olive"]
hist_labels = ["Background", "Signal"]
hist_colors = ["b", "r"]

keys = ['mjj', 'j1_features', 'j2_features']

include_images =  any(np.array(model_type) == 1)
if(include_images): 
    x_label = j_label + "_images"
    print("including images")
    keys.append(x_label)

#keys = ["j1_images", "j2_images", "j1_features", "j2_features"]

if(single_file):
    num_data = -1
    data_start = 0
    data = DataReader(fin=fin, sig_idx = sig_idx, data_start = data_start, data_stop = data_start + num_data, keys = keys, keep_mlow = m_low, keep_mhigh = m_high, 
            hadronic_only = hadronic_only, d_eta = d_eta, batch_start = batch_start, batch_stop = batch_stop, sig_frac = sig_frac)
    data.read()
    if(include_images): images = data[x_label]
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
mjj = data['mjj']

#sig_region_events = (mjj > msig_low) & (mjj < msig_high)
#bkg_region_events = (sig_region_events ==0)
if(include_images): images = data[j_label + "_images"]
else: images = None
dense_inputs = data[j_label + "_features"]


model_scores = []
for idx,f in enumerate(f_models):

    scores = get_single_jet_scores(model_dir + f_models[idx], model_type[idx], j_images = images, j_dense_inputs = dense_inputs, 
            num_models = num_ensemble[idx], batch_size = 512)
    #hist_scores = [scores[bkg_events], scores[sig_events]]
    #make_histogram(hist_scores, hist_labels, hist_colors, 'Labeler Score', "", 100,
            #normalize = True, save = True, fname = plot_dir + f.replace('.h5', '').replace("/", "") +"_scores.png")
    model_scores.append(scores)
    #eff_cut1 = compute_effcut_metric(scores[sig_region_events], scores[bkg_region_events], eff = 0.01)
    #eff_cut2 = compute_effcut_metric(scores[sig_region_events], scores[bkg_region_events], eff = 0.02)
    #eff_cut3 = compute_effcut_metric(scores[sig_region_events], scores[bkg_region_events], eff = 0.03)
    #print(eff_cut3, eff_cut2, eff_cut1)




make_roc_curve(model_scores, Y, labels = labels, colors = colors,  logy = True, fname=plot_dir + roc_plot_name)
make_sic_curve(model_scores, Y, labels = labels, colors = colors, ymax = 10.,  fname=plot_dir + sic_plot_name)
del data
