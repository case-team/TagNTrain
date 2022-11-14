import sys
sys.path.append('..')
from utils.TrainingUtils import *
import h5py

#fin = "../data/BB_v2_2500_images/BB_images_testset.h5"
#model_dir = "../models/BB_v2_M2500/"
#options.output = "../plots/corr_test_m3500_ptsort/"
#
#os.system("mkdir %s" % options.output)
#
#j1_classifier = "j1_autoencoder_m2500.h5"
#j2_classifier = "j2_autoencoder_m2500.h5"
#options.label = "corr_"
#is_auto_encoder = True
#
#pt_sort = True
#rand_sort = False
#
#sig_idx = 1
#
#num_data = -1
#data_start = 0
#
##sig_frac = 0.00081
#sig_frac = 0.0
#m_low = 3150
#m_high = 3850
#
#
#data = DataReader(fin, signal_idx = sig_idx, start = data_start, stop = data_start + num_data, keys = keys, m_low = m_low, m_high = m_high )
#data.read()

parser = input_options()
options = parser.parse_args()

if(options.output[-1] != '/'): options.output +='/'

os.system("mkdir %s" %options.output)
options.keys = ['mjj', 'j1_images', 'j2_images', 'jet_kinematics', 'j1_features', 'j2_features']

data, _ = load_dataset_from_options(options)

j1_images = data['j1_images']
j2_images = data['j2_images']
mjj = data['mjj']
j1_pt = data['jet_kinematics'][:,2]
j2_pt = data['jet_kinematics'][:,6]

j1_features = data['j1_features']
j2_features = data['j2_features']

j1_m = j1_features[:,0]
j2_m = j2_features[:,0]

Y = data['label'].reshape(-1)
sig_events = (Y > 0.9)
bkg_events = (Y < 0.1)


j1_model = tf.keras.models.load_model(options.labeler_name)
j2_model = tf.keras.models.load_model(options.labeler_name)

j1_scores = data.labeler_scores(j1_model,  'j1_images')
j2_scores = data.labeler_scores(j2_model,  'j2_images')

#if(pt_sort):
#    swapping_idxs = j2_pt > j1_pt
#elif(rand_sort):
#    swapping_idxs = np.random.choice(a=[True,False], size = j1_scores.shape[0])
#else:
#    swapping_idxs = np.zeros_like(j1_scores, dtype=np.bool8)
#
#j1sorted_scores = np.copy(j1_scores)
#j2sorted_scores = np.copy(j2_scores)
#j1sorted_scores[swapping_idxs] = j2_scores[swapping_idxs]
#j2sorted_scores[swapping_idxs] = j1_scores[swapping_idxs]




#j1sorted_m = np.copy(j1_m)
#j2sorted_m = np.copy(j2_m)
#j1sorted_m[swapping_idxs] = j2_m[swapping_idxs]
#j2sorted_m[swapping_idxs] = j1_m[swapping_idxs]
#
#
#j1sorted_pt = np.copy(j1_pt)
#j2sorted_pt = np.copy(j2_pt)
#j1sorted_pt[swapping_idxs] = j2_pt[swapping_idxs]
#j2sorted_pt[swapping_idxs] = j1_pt[swapping_idxs]
#
#
#j1_features = data['j1_features'][()]
#j2_features = data['j2_features'][()]
#j1sorted_features = np.copy(j1_features)
#j2sorted_features = np.copy(j2_features)
#j1sorted_features[swapping_idxs] = j2_features[swapping_idxs]
#j2sorted_features[swapping_idxs] = j1_features[swapping_idxs]



#print("idxs", swapping_idxs[:5])
print("pts", j1_pt[:5], j2_pt[:5])
print("ms", j1_m[:5], j2_m[:5])
print("scores", j1_scores[:5], j2_scores[:5])



correlation = np.corrcoef(j1_m, j2_m)[0,1]
print("Mass correlation :", correlation)

dijet_mass = []


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
print("Score correlation :", correlation)
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
plt.savefig(options.output + options.label + "correlation.png")

make_scatter_plot(bkg_j1s, j1_m[bkg_events], 'b', ['J1 Score', 'J1 Mass (GeV)'], fname = options.output + options.label + "j1_ae_m_corr.png")
make_scatter_plot(bkg_j2s, j2_m[bkg_events], 'b', ['J2 Score', 'J2 Mass (GeV)'], fname = options.output + options.label + "j2_ae_m_corr.png")
make_scatter_plot(bkg_j1s, j1_pt[bkg_events], 'b', ['J1 Score', 'J1 pT (GeV)'], fname = options.output + options.label + "j1_ae_pt_corr.png")
make_scatter_plot(bkg_j2s, j2_pt[bkg_events], 'b', ['J2 Score', 'J2 pT (GeV)'], fname = options.output + options.label + "j2_ae_pt_corr.png")
make_scatter_plot(j1_m[bkg_events], j2_m[bkg_events], 'b', ['J1 Mass (GeV)', 'J2 Mass (GeV)'], fname = options.output + options.label + "j1_m_j2_m_corr.png")
make_scatter_plot(j1_features[:,1][bkg_events],j2_features[:,1][bkg_events], 'b', ['J1 Tau1', 'J2 Tau1'], fname = options.output + options.label + "j1_tau1_j2_tau1_corr.png")
make_scatter_plot(j1_features[:,2][bkg_events], j2_features[:,2][bkg_events], 'b', ['J1 Tau2', 'J2 Tau2'], fname = options.output + options.label + "j1_tau2_j2_tau2_corr.png")
make_scatter_plot(j1_m[bkg_events], j1_pt[bkg_events], 'b', ['J1 Mass', 'J1 pT (GeV)'], fname = options.output + options.label + "j1_m_pt_corr.png")
make_scatter_plot(j2_m[bkg_events], j2_pt[bkg_events], 'b', ['J2 Mass', 'J2 pT (GeV)'], fname = options.output + options.label + "j2_m_pt_corr.png")
make_scatter_plot(j1_pt[bkg_events], j2_pt[bkg_events], 'b', ['J1 pT', 'J2 pT'], fname = options.output + options.label + "j1_pt_j2_pt_corr.png")
make_scatter_plot(j1_pt[bkg_events], j1_features[:,1][bkg_events], 'b', ['J1 pT', 'J1 Tau1'], fname = options.output + options.label + "j1_pt_tau1_corr.png")
make_scatter_plot(j2_pt[bkg_events], j2_features[:,1][bkg_events], 'b', ['J2 pT', 'J2 Tau1'], fname = options.output + options.label + "j2_pt_tau1_corr.png")
make_scatter_plot(j1_pt[bkg_events], j1_features[:,2][bkg_events], 'b', ['J1 pT', 'J1 Tau2'], fname = options.output + options.label + "j1_pt_tau2_corr.png")
make_scatter_plot(j2_pt[bkg_events], j2_features[:,2][bkg_events], 'b', ['J2 pT', 'J2 Tau2'], fname = options.output + options.label + "j2_pt_tau2_corr.png")
