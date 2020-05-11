import sys
sys.path.append('..')
from utils.TrainingUtils import *
import fastjet as fj
import energyflow as ef
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential, load_model
from energyflow.utils import data_split, pixelate, standardize, to_categorical, zero_center
import h5py


fin = "../data/jet_images.h5"
plot_dir = "../plots/"
model_dir = "../models/"

j1_classifier = "j1_pure_cwola.h5"
j2_classifier = "j2_pure_cwola.h5"
plot_label = "pure_cwola_"

#j1_classifier = "j1_TNT1_CNN_no_mjj_s10p_1p.h5"
#j2_classifier = "j2_TNT1_CNN_no_mjj_s10p_1p.h5"
#plot_label = "TNT1_no_mjj_"

is_auto_encoder = False

num_data = 100000
data_start = 1000000

filt_sig = True
sig_frac = 0.01



hf_in = h5py.File(fin, "r")

j1_images = hf_in['j1_images'][data_start:data_start + num_data]
j1_images = np.expand_dims(j1_images, axis=-1)
j1_images = standardize(*zero_center(j1_images, np.zeros_like(j1_images)))[0]
j2_images = hf_in['j2_images'][data_start:data_start + num_data]
j2_images = np.expand_dims(j2_images, axis=-1)
j2_images = standardize(*zero_center(j2_images, np.zeros_like(j2_images)))[0]

jet_infos = hf_in['jet_infos'][data_start:data_start + num_data]
Y = jet_infos[:,0] #is signal bit is first bit of info



#force signal to be a given fraction of total events because input dataset signal is too big (10%)
if(filt_sig):

    print("Filtering sig to be %.3f" % sig_frac)
    train_mask = get_signal_mask(Y, sig_frac)
    Y = Y[train_mask]
    j1_images =  j1_images[train_mask]
    j2_images =  j2_images[train_mask]


j1_model = load_model(model_dir + j1_classifier)
j2_model = load_model(model_dir + j2_classifier)

j1_scores = j1_model.predict(j1_images, batch_size = 1000)
j2_scores = j2_model.predict(j2_images, batch_size = 1000)

if(is_auto_encoder):
    j1_scores =  np.mean(keras.losses.mean_squared_error(j1_scores, j1_images), axis=(1,2)).reshape(-1)
    j2_scores =  np.mean(keras.losses.mean_squared_error(j2_scores, j2_images), axis=(1,2)).reshape(-1)

j1_scores = j1_scores.reshape(-1)
j2_scores = j2_scores.reshape(-1)

dijet_mass = []

sig_events = (Y > 0.9)
bkg_events = (Y < 0.1)

sig_j1s = j1_scores[sig_events]
sig_j2s = j2_scores[sig_events]

bkg_j1s = j1_scores[bkg_events]
bkg_j2s = j2_scores[bkg_events]


save_figs = True
labels = ['background', 'signal']
colors = ['b', 'r']
alpha = 0.5
size = 0.4

fig, ax = plt.subplots()
ax.scatter(bkg_j1s, bkg_j2s , alpha = alpha, c = colors[0], s=size, label = "background")
ax.scatter(sig_j1s, sig_j2s , alpha = alpha, c = colors[1], s=size, label = "signal")

#m_cov = np.cov(bkg_j1s, bkg_j2s)
#correlation = m_cov[0,1] / np.sqrt(m_cov[0,0] * m_cov[1,1])
correlation = np.corrcoef(bkg_j1s, bkg_j2s)[0,1]
print("correlation :", correlation)
text_str = r'$\rho_{j1,j2} $ = %.3f' % correlation
print(text_str)
plt.text(0.45, 0.45, text_str, fontsize=14)
#ax.legend(loc='upper right')

plt.xlim([0.1, 0.6])
plt.ylim([0.1, 0.5])
ax.set_ylabel("Jet2 Score", fontsize=16)
ax.set_xlabel("Jet1 Score", fontsize=16)
plt.tick_params(axis='y', labelsize=12)
plt.tick_params(axis='x', labelsize=12)
plt.savefig(plot_dir + plot_label + "correlation.png")



