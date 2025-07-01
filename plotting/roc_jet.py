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
roc_plot_name = "%s_roc_%s.png" %(j_label, options.label)
sic_plot_name = "%s_sic_%s.png" % (j_label, options.label)

options.no_minor_bkgs = True


single_file = True
options.hadronic_only = True

model_dir = "../models/"

#model type: 0 is CNN, 1 is autoencoder, 2 is dense network, 5 is VAE

f_models = [
        
        "AEs/jrand_autoencoder_m2500.h5",
        "../models/AEs/AEs_sep1/jrand_AE_kfold0_mbin13.h5",
        #"AEs/AEs_sep1/jrand_AE_kfold0_mbin3.h5",
        #"AEs/AEs_sep1/jrand_AE_kfold0_mbin4.h5",
        #"AEs/AEs_sep1/jrand_AE_kfold0_mbin5.h5",
        #"AEs/AEs_sep1/jrand_AE_kfold0_mbin6.h5",
        #"AEs/AEs_sep1/jrand_AE_kfold1_mbin6.h5",
        #"AEs/AEs_sep1/jrand_AE_kfold2_mbin6.h5",
        #"AEs/AEs_data_SR_june9/jrand_AE_kfold0_mbin3.h5",
        #"AEs/AEs_data_SR_june9/jrand_AE_kfold0_mbin4.h5",
        #"AEs/AEs_data_SR_june9/jrand_AE_kfold0_mbin5.h5",
        #"AEs/AEs_data_SR_june9/jrand_AE_kfold0_mbin6.h5",
        #"AEs/AEs_data_SR_june9/jrand_AE_kfold1_mbin6.h5",
        #"AEs/AEs_data_SR_june9/jrand_AE_kfold2_mbin6.h5",

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
        "AE jrand m2500",
        "AE kfold0",
        #"AE mbin3",
        #"AE mbin4",
        #"AE mbin5",
        #"AE mbin6 k0",
        #"AE mbin6 k1",
        #"AE mbin6 k2",
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

data, _ = load_dataset_from_options(options)
sig_only_data = load_signal_file(options)

if(include_images): 
    bkg_images = data[j_label + "_images"]
    sig_images = sig_only_data[j_label + "_images"]
    images = np.concatenate([sig_images, bkg_images])
else: images = None

bkg_dense_inputs = data[j_label + "_features"]
sig_dense_inputs = sig_only_data[j_label + "_features"]
dense_inputs = np.concatenate([sig_dense_inputs, bkg_dense_inputs])

nsig = sig_dense_inputs.shape[0]
nbkg = bkg_dense_inputs.shape[0]

Y = np.concatenate([np.ones(nsig), np.zeros(nbkg)]).reshape(-1)

bkg_events = Y < 0.1
sig_events = Y > 0.9
mjj = data['mjj']

#sig_region_events = (mjj > msig_low) & (mjj < msig_high)
#bkg_region_events = (sig_region_events ==0)

if(not os.path.exists(options.output)): os.system("mkdir %s" % options.output)

model_scores = []
for idx,f in enumerate(f_models):
    print(f)

    scores = get_single_jet_scores(model_dir + f_models[idx], model_type[idx], j_images = images, j_dense_inputs = dense_inputs, 
            num_models = num_models[idx], batch_size = 512)
    model_scores.append(scores)

    #hist_scores = [scores[bkg_events], scores[sig_events]]
    #make_histogram(hist_scores, hist_labels, hist_colors, 'Labeler Score', "", 100,
            #normalize = True, save = True, fname = options.output + f.replace('.h5', '').replace("/", "") +"scores.png")

    #eff_cut1 = compute_effcut_metric(scores[sig_region_events], scores[bkg_region_events], eff = 0.01)
    #eff_cut2 = compute_effcut_metric(scores[sig_region_events], scores[bkg_region_events], eff = 0.02)
    #eff_cut3 = compute_effcut_metric(scores[sig_region_events], scores[bkg_region_events], eff = 0.03)
    #print(eff_cut3, eff_cut2, eff_cut1)




make_roc_curve(model_scores, Y, labels = labels, colors = colors,  logy = True, fname=options.output + roc_plot_name)
make_sic_curve(model_scores, Y, labels = labels, colors = colors, ymax = 3.0,  fname=options.output + sic_plot_name)
del data
