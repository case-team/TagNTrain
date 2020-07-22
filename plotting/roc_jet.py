import sys
sys.path.append('..')
from utils.TrainingUtils import *
import h5py


fin = "../data/BB_images_v1.h5"
#f_sig = "../data/BulkGravToZZToZhadZhad_narrow_M-2500.h5" #1
#f_sig = "../data/WprimeToWZToWhadZhad_narrow_M-3500.h5" #2
f_sig = "../data/WkkToWRadionToWWW_M2500-R0-08.h5" #3
f_bkg = "../data/QCD_only.h5"
j_label = "j2_"
sig_idx = 3
plot_name = "%ss%i_roc.png" %(j_label, sig_idx)

m_low = 2250.
m_high = 2750.
#m_low = 3200.
#m_high = 3800.

single_file = False

plot_dir = "../plots/BB1_test_june/"
model_dir = "../models/BB1_test_june/"

#model type: 0 is CNN, 1 is autoencoder, 2 is dense network

#f_models = ["j1_autoencoder.h5",  "j1_autoencoder_s%i.h5", "j1_cwola_hunting_s%i.h5", "j1_TNT0_s%i.h5","j1_TNT0_s%i_v2.h5"]
f_models = ["j2_autoencoder.h5",  "j2_autoencoder_s%i.h5", "j2_cwola_hunting_s%i.h5", "j2_TNT0_s%i.h5","j2_TNT0_s%i_v2.h5"]
labels = ["Auto Encoder (train all)", "Auto Encoder (train sidebands)","CWoLa Hunting (One Jet)", "TNT (AE)", "TNT (CWoLa)"] #, "Cwola Hunting (1 Jet)"]
model_type = [1, 1, 0, 0, 0, 0, 0, 0,0,0]


colors = ["g", "b", "r", "gray", "purple", "pink", "orange", "m", "skyblue", "yellow"]
hist_labels = ["Background", "Signal"]
hist_colors = ["b", "r"]

keys = ['mjj', 'j1_images', 'j2_images']
x_label = j_label + "images"

if(single_file):
    num_data = 400000
    data_start = 20000000
    data = DataReader(fin, signal_idx = sig_idx, start = data_start, stop = data_start + num_data, keys = keys, m_low = m_low, m_high = m_high )
    data.read()
    images = data[x_label]
    Y = data['label']

else:
    bkg_start = 1000000
    n_bkg = 400000
    sig_start = 20000
    sig_stop = -1
    d_bkg = DataReader(f_bkg, keys = keys, signal_idx = -1, start = bkg_start, stop = bkg_start + n_bkg, m_low = m_low, m_high = m_high )
    d_bkg.read()
    d_sig = DataReader(f_sig, keys = keys, signal_idx = -1, start = sig_start, stop = sig_stop, m_low = m_low, m_high = m_high )
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
    model = tf.keras.models.load_model(model_dir + f)
    print(idx, f, model_type[idx])
    if(model_type[idx] == 0): scores = model.predict(images, batch_size = 500)
    elif(model_type[idx] == 2): scores = model.predict(dense_inputs, batch_size = 500)
    else:
        reco_images = model.predict(images, batch_size=500)
        scores = np.mean(np.square(images - reco_images), axis = (1,2)).reshape(-1)
    scores = scores.reshape(-1)

    hist_scores = [scores[bkg_events], scores[sig_events]]
    make_histogram(hist_scores, hist_labels, hist_colors, 'Labeler Score', "", 100,
            normalize = True, save = True, fname = plot_dir + f[:-3] +"_scores.png")
    model_scores.append(scores)


roc_plt_name  = plot_dir + plot_name
make_roc_curve(model_scores, Y, labels = labels, colors = colors, save = True, logy=True, fname=roc_plt_name)
