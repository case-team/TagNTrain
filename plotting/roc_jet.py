import sys
sys.path.append('..')
from utils.TrainingUtils import *
import h5py

parser = input_options()
options = parser.parse_args()

compute_mjj_window(options)
options.keep_mlow = options.mjj_low
options.keep_mhigh = options.mjj_high

j_label = "j1"
fj_label = "jrand"
roc_plot_name = "%s_roc_%s.png" %(j_label, options.label)
sic_plot_name = "%s_sic_%s.png" % (j_label, options.label)

options.no_minor_bkgs = True


single_file = True
options.hadronic_only = True

model_dir = "../models/"

#model type: 0 is CNN, 1 is autoencoder, 2 is dense network, 5 is VAE

f_models = [
        
        "AEs/AEs_sep1/jrand_AE_kfold0_mbin3.h5",
        "AEs/AEs_sep1/jrand_AE_kfold0_mbin4.h5",
        "AEs/AEs_sep1/jrand_AE_kfold0_mbin5.h5",
        "AEs/AEs_sep1/jrand_AE_kfold0_mbin6.h5",
        "AEs/AEs_sep1/jrand_AE_kfold1_mbin6.h5",
        "AEs/AEs_sep1/jrand_AE_kfold2_mbin6.h5",
        #"dense_AE_test/AE_mbin13_latent1.h5",
        #"dense_AE_test/AE_mbin13_latent2.h5",
        #"dense_AE_test/AE_mbin13_latent3.h5",
        #"AEs_may31/jrand_AE_kfold0_mbin13.h5",
        #"j2_supervised_Wp.h5",
        #"j2_supervised_Wp_gaus_norm.h5",
        #"j2_supervised_Wp_uniform_norm.h5",
        #"j2_cwola_Wp.h5",
        #"j2_cwola_Wp_gaus_norm.h5",
        #"j2_cwola_Wp_uniform_norm.h5",
        #"j1_cwola_Wp_spb10.h5",
        #"j1_cwola_Wp_spb10_gaus_norm.h5",
        #"j1_cwola_Wp_spb10_uniform_norm.h5",

#'test_spb30.h5',
#'test_batch250.h5',
#'test_batch1k.h5',
#'test_batch5k.h5',
#'test_spb80.h5',

        ] 
#model_type = [-1,-1,-1,1] 
model_type = [1,1,1,1,1,1] 
#model_type = [2,2,2,2,2,2] 
#num_models = [5,5,5,5,5,5]
num_models = [1,1,1,1,1,1]
labels = [
        #'Dense AE (latent size 1)',
        #'Dense AE (latent size 2)',
        #'Dense AE (latent size 3)',
        #'CNN AE (latent size 6)',
        #"supervised",
        #"supervised gaus norm",
        #"supervised uniform norm",
        #"cwola",
        #"cwola gaus norm",
        #"cwola uniform norm",

        #"cwola spb10",
        #"cwola spb10 gaus norm",
        #"cwola spb10 uniform norm"
        "AE mbin3",
        "AE mbin4",
        "AE mbin5",
        "AE mbin6 k0",
        "AE mbin6 k1",
        "AE mbin6 k2",
        ]



colors = ["g", "b", "r", "gray", "purple", "pink", "orange", "m", "skyblue", "yellow", "lightcoral", "gold","olive"]
hist_labels = ["Background", "Signal"]
hist_colors = ["b", "r"]

options.keys = ['mjj', 'j1_features', 'j2_features']

include_images =  any(np.array(model_type) == 1)
if(include_images): 
    x_label = j_label + "_images"
    print("including images")
    options.keys.append(x_label)

#keys = ["j1_images", "j2_images", "j1_features", "j2_features"]

if(single_file):
    num_data = -1
    data, _ = load_dataset_from_options(options)
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
    print(f)

    dense_inputs_norm = dense_inputs
    if("uniform" in f):
        print("Uniform")
        qt = create_transforms(dense_inputs, dist = 'uniform')
        dense_inputs_norm = qt.transform(dense_inputs)
    if("gaus" in f):
        print("gaus")
        qt = create_transforms(dense_inputs, dist = 'normal')
        dense_inputs_norm = qt.transform(dense_inputs)


    scores = get_single_jet_scores(model_dir + f_models[idx], model_type[idx], j_images = images, j_dense_inputs = dense_inputs_norm, 
            num_models = num_models[idx], batch_size = 512)
    #hist_scores = [scores[bkg_events], scores[sig_events]]
    #make_histogram(hist_scores, hist_labels, hist_colors, 'Labeler Score', "", 100,
            #normalize = True, save = True, fname = plot_dir + f.replace('.h5', '').replace("/", "") +"_scores.png")
    model_scores.append(scores)
    #eff_cut1 = compute_effcut_metric(scores[sig_region_events], scores[bkg_region_events], eff = 0.01)
    #eff_cut2 = compute_effcut_metric(scores[sig_region_events], scores[bkg_region_events], eff = 0.02)
    #eff_cut3 = compute_effcut_metric(scores[sig_region_events], scores[bkg_region_events], eff = 0.03)
    #print(eff_cut3, eff_cut2, eff_cut1)




make_roc_curve(model_scores, Y, labels = labels, colors = colors,  logy = True, fname=options.output + roc_plot_name)
make_sic_curve(model_scores, Y, labels = labels, colors = colors, ymax = 10.,  fname=options.output + sic_plot_name)
del data
