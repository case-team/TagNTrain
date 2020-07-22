import sys
sys.path.append('..')
from utils.TrainingUtils import *
import h5py

fin = "../data/BB_images_v1.h5"
plot_dir = "../plots/BB1_test_june/"
model_dir = "../models/BB1_test_june/"

j1_classifier = "j1_autoencoder_sm_s3.h5"
j2_classifier = "j2_autoencoder_sm_s3.h5"
plot_label = "autoencoder_sm_s3_corr_"
is_auto_encoder = True

sig_idx = 1

num_data = 400000
data_start = 0

sig_frac = 0.00081
m_low = 2250
m_high = 2750


keys = ['mjj', 'j1_images', 'j2_images', 'jet_kinematics']
data = DataReader(fin, signal_idx = sig_idx, start = data_start, stop = data_start + num_data, keys = keys, m_low = m_low, m_high = m_high )
data.read()
j1_images = data['j1_images']
j2_images = data['j2_images']
mjj = data['mjj']
j1_m = data['jet_kinematics'][:,5]
j2_m = data['jet_kinematics'][:,9]
j1_pt = data['jet_kinematics'][:,2]
j2_pt = data['jet_kinematics'][:,6]
Y = data['label'].reshape(-1)


j1_model = tf.keras.models.load_model(model_dir + j1_classifier)
j2_model = tf.keras.models.load_model(model_dir + j2_classifier)

j1_scores = j1_model.predict(j1_images, batch_size = 1000)
j2_scores = j2_model.predict(j2_images, batch_size = 1000)

if(is_auto_encoder):
    j1_scores =  np.mean(np.square(j1_scores - j1_images), axis=(1,2))
    j2_scores =  np.mean(np.square(j2_scores -  j2_images), axis=(1,2))

j1_scores = j1_scores.reshape(-1)
j2_scores = j2_scores.reshape(-1)
print(np.amax(j1_scores), np.amax(j2_scores))

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
#ax.scatter(sig_j1s, sig_j2s , alpha = alpha, c = colors[1], s=size, label = "signal")

#m_cov = np.cov(bkg_j1s, bkg_j2s)
#correlation = m_cov[0,1] / np.sqrt(m_cov[0,0] * m_cov[1,1])
correlation = np.corrcoef(bkg_j1s, bkg_j2s)[0,1]
print("correlation :", correlation)
text_str = r'$\rho_{j1,j2} $ = %.3f' % correlation
plt.annotate(text_str, xy = (0.05, 0.95), xycoords = 'axes fraction', fontsize=14)
print(text_str)
plt.text(0.45, 0.45, text_str, fontsize=14)
#ax.legend(loc='upper right')

ax.set_ylabel("Jet2 Score", fontsize=16)
ax.set_xlabel("Jet1 Score", fontsize=16)
plt.xlim([0., 0.0008])
plt.ylim([0., 0.0008])
plt.tick_params(axis='y', labelsize=12)
plt.tick_params(axis='x', labelsize=12)
plt.savefig(plot_dir + plot_label + "correlation.png")

make_scatter_plot(bkg_j1s, j1_m[bkg_events], 'b', ['J1 Score', 'J1 Mass (GeV)'], fname = plot_dir + plot_label + "j1_m_corr.png")
make_scatter_plot(bkg_j2s, j2_m[bkg_events], 'b', ['J2 Score', 'J2 Mass (GeV)'], fname = plot_dir + plot_label + "j2_m_corr.png")
make_scatter_plot(bkg_j1s, j1_pt[bkg_events], 'b', ['J1 Score', 'J1 pT (GeV)'], fname = plot_dir + plot_label + "j1_pt_corr.png")
make_scatter_plot(bkg_j2s, j2_pt[bkg_events], 'b', ['J2 Score', 'J2 pT (GeV)'], fname = plot_dir + plot_label + "j2_pt_corr.png")

