import sys
sys.path.append('..')
from utils.TrainingUtils import *
import h5py


fin = "../data/BB_v2_2500_images/BB_images_testset.h5"
#norm_img = "../data/jet_images_BB2500_norm.h5"
norm_img = ""
j_label = "j1_"
sig_idx = 1
plot_name = "%ss%i_norm_roc.png" %(j_label, sig_idx)

m_low = 2250.
m_high = 2750.
#m_low = 3150.
#m_high = 3850.

single_file = True
hadronic_only = True

plot_dir = "../plots/BB_v2_M2500/"
model_dir = "../models/BB_v2_M2500/"

#model type: 0 is CNN, 1 is autoencoder, 2 is dense network, 5 is VAE

#f_models = ["j1_autoencoder_s1.h5",  "j1_vae_s1/",  "j1_vae_all_s1/"] 
#f_models = ["j2_autoencoder_m2500.h5",  "j2_vae_m2500/",  "j2_vae_all/"] 
model_type = [1, 5, 5] 
labels = ["AE (train SB)", "VAE (train SB)", "VAE (train all)"]


colors = ["g", "b", "r", "gray", "purple", "pink", "orange", "m", "skyblue", "yellow", "lightcoral", "gold","olive"]
hist_labels = ["Background", "Signal"]
hist_colors = ["b", "r"]

keys = ['mjj', 'j1_images', 'j2_images']
x_label = j_label + "images"

if(single_file):
    num_data = -1
    data_start = 0
    data = DataReader(fin, signal_idx = sig_idx, start = data_start, stop = data_start + num_data, keys = keys, m_low = m_low, m_high = m_high, hadronic_only = hadronic_only, 
            norm_img = norm_img)
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
    d_bkg = DataReader(f_bkg, keys = keys, signal_idx = -1, start = bkg_start, stop = bkg_start + n_bkg, m_low = m_low, m_high = m_high, hadronic_only = hadronic_only )
    d_bkg.read()
    d_sig = DataReader(f_sig, keys = keys, signal_idx = -1, start = sig_start, stop = sig_stop, m_low = m_low, m_high = m_high, hadronic_only = hadronic_only )
    d_sig.read()

    im_bkg = d_bkg[x_label]
    im_sig = d_sig[x_label]
    images = np.concatenate((im_bkg, im_sig), axis = 0)
    Y = np.concatenate((np.zeros(im_bkg.shape[0], dtype=np.int8), np.ones(im_sig.shape[0], dtype=np.int8)))

bkg_events = Y < 0.1
sig_events = Y > 0.9


model_scores = []
for idx,f in enumerate(f_models):
    if('%' in f): f = f % sig_idx
    if(model_type[idx] <5):
        model = tf.keras.models.load_model(model_dir + f)
        print(idx, f, model_type[idx])
        if(model_type[idx] == 0): scores = model.predict(images, batch_size = 1000)
        elif(model_type[idx] == 2): scores = model.predict(dense_inputs, batch_size = 1000)
        else:
            reco_images = model.predict(images, batch_size=500)
            scores = np.mean(np.square(images - reco_images), axis = (1,2)).reshape(-1)
            print_image(images[0])
            print("reco:")
            print_image(reco_images[0])
            #print(np.mean(scores))
        scores = scores.reshape(-1)

    else:
        model = VAE(0, model_dir = model_dir + f)
        model.load()
        reco_images, z_mean, z_log_var = model.predict_with_latent(images)
        print_image(images[0])
        print("reco:")
        print_image(reco_images[0])
        scores = compute_loss_of_prediction_mse_kl(images, reco_images, z_mean, z_log_var)[0]
        print(scores.shape)
        scores.reshape(-1)
    hist_scores = [scores[bkg_events], scores[sig_events]]
    make_histogram(hist_scores, hist_labels, hist_colors, 'Labeler Score', "", 100,
            normalize = True, save = True, fname = plot_dir + f.replace('.h5', '').replace("/", "") +"_scores.png")
    model_scores.append(scores)




roc_plt_name  = plot_dir + plot_name
make_roc_curve(model_scores, Y, labels = labels, colors = colors, save = True, logy=True, fname=roc_plt_name)
del data
