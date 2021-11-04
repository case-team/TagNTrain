import sys
sys.path.append('..')
from utils.TrainingUtils import *
import matplotlib.colors as mcolors
from optparse import OptionParser
from optparse import OptionGroup

parser = input_options()
options = parser.parse_args()

fin = options.fin
plot_dir = options.plot_dir
model_dir = options.model_dir



model_type = options.model_type
if(len(options.label)> 0 and options.label[-1] != '_'):
    options.label += '_'

num_data = options.num_data
data_start = options.data_start

use_or = False
use_j = 0

sample_standardize = False

mjj_window = 500.




js_threshholds = [70., 80., 90., 93., 95., 97.]
jj_threshholds = [75., 90., 95., 99., 99.7]


use_dense = (model_type == 2 or model_type == 4)

    

if(options.labeler_name != ""):
    options.keys = ['j1_features', 'j2_features', 'j1_images', 'j2_images', 'jet_kinematics']
else: 
    options.keys = ['jet_kinematics']

j1_images = j2_images = jj_images = j1_dense_inputs = j2_dense_inputs = jj_dense_inputs = None

data, _ = load_dataset_from_options(options)
Y = data['label'].reshape(-1)
mjj = data['jet_kinematics'][:,0]
j1_m = data['jet_kinematics'][:,5]
j2_m = data['jet_kinematics'][:,9]

if(options.labeler_name != ""):
    j1_images = data['j1_images']
    j2_images = data['j2_images']



batch_size = 1000



save_figs = True
labels = ['background', 'signal']
#colors = ['b', 'r', 'g', 'purple', 'pink', 'black', 'magenta', 'pink']
colors = ["g", "gray", "b", "r","m", "skyblue", "pink"]
colors_sculpt = []
colormap = cm.viridis
normalize = mcolors.Normalize(vmin = 0., vmax=100.)



n_m_bins = 30
m_range = (1200., 7000.)



sig_events = (Y > 0.9)
bkg_events = (Y < 0.1)
minor_bkg_events = (Y < -0.5)
mjj_dists = []
j1_pt_dists = []
j2_pt_dists = []
dist_labels = []
#mjj_dists.append(mjj[bkg_events])

#No selection
mjj_dists.append(mjj[bkg_events])
make_outline_hist([mjj[bkg_events]],  mjj[sig_events], labels, ['b','r'], 'Dijet Mass (GeV)', "No Selection", n_m_bins, logy = True,
         save = save_figs,  h_range = m_range, fname=plot_dir + options.label + "nocut_mass.png" )



mjj_sig = np.mean(mjj[sig_events])
m_low = options.mjj_low
m_high = options.mjj_high
if(m_low< 0):
    m_low = mjj_sig - mjj_window/2.
if(m_high< 0):
    m_high = mjj_sig + mjj_window/2.

in_window = (mjj > m_low) & (mjj < m_high)
S = mjj[sig_events & in_window].shape[0]
B = mjj[bkg_events & in_window].shape[0]
minor_B = mjj[minor_bkg_events & in_window].shape[0]

eff_S_in_window = S / mjj[sig_events].shape[0]

overall_S = mjj[sig_events].shape[0]
overall_B = mjj[bkg_events].shape[0]

print("Mean signal mjj is %.0f" % mjj_sig);
print("Mjj window %f to %f " % (m_low, m_high))
print("Before selection: %i signal events and %i bkg events in mjj window (%i minor bkg)" % (S,B, minor_B))
print("Frac. of signal in mjj window is %.3f " % eff_S_in_window)
print("S/B %f, sigificance ~ %.1f " % (float(S)/B, S/np.sqrt(B)))
print("Sig frac (overall) %f " % (float(overall_S)/overall_B))



if(options.labeler_name != ""): #apply selection

    print("Using model %s" % f)
    if(options.model_type <= 2 or options.model_type == 5): #classifier on each jet

        threshholds = js_threshholds
        #j1_score, j2_score = get_jet_scores(model_dir, f, options.model_type, j1rand_images, j2rand_images, j1rand_dense_inputs, j2rand_dense_inputs, num_models = options.num_models)
        j1_score, j2_score = get_jet_scores("", f, options.model_type, j1_images, j2_images, j1_dense_inputs, j2_dense_inputs, num_models = options.num_models)
        j1_cut_vals = [np.percentile(j1_scores, thresh) for thresh in threshholds]
        j2_cut_vals = [np.percentile(j2_scores, thresh) for thresh in threshholds]
    else:
        threshholds = jj_threshholds
        jj_scores = get_jj_scores("", f, options.model_type, jj_images, jj_dense_inputs, num_models = options.num_models)
        jj_cut_vals = [np.percentile(jj_scores, thresh) for thresh in jj_threshholds]

    #if(model_type <= 2):
    #    threshholds = js_threshholds
    #    j1_model = load_model(model_dir + "j1_" + labeler_name)
    #    j2_model = load_model(model_dir + "j2_" + labeler_name)
    #    if(model_type == 0):
    #        j1_scores = j1_model.predict(j1_images, batch_size = batch_size)
    #        j2_scores = j2_model.predict(j2_images, batch_size = batch_size)
    #    elif(model_type ==1):
    #        j1_reco_images = j1_model.predict(j1_images, batch_size = batch_size)
    #        j2_reco_images = j2_model.predict(j2_images, batch_size = batch_size)
    #        j1_scores =  np.mean(keras.losses.mean_squared_error(j1_reco_images, j1_images), axis=(1,2)).reshape(-1)
    #        j2_scores =  np.mean(keras.losses.mean_squared_error(j2_reco_images, j2_images), axis=(1,2)).reshape(-1)
    #    elif(model_type == 2):
    #        j1_scores = j1_model.predict(j1_dense_inputs, batch_size = batch_size)
    #        j2_scores = j2_model.predict(j2_dense_inputs, batch_size = batch_size)

    #    j1_scores = j1_scores.reshape(-1)
    #    j2_scores = j2_scores.reshape(-1)
    #    j1_cut_vals = [np.percentile(j1_scores, thresh) for thresh in threshholds]
    #    j2_cut_vals = [np.percentile(j2_scores, thresh) for thresh in threshholds]
    #else:
    #    threshholds = jj_threshholds
    #    if(model_type == 3): #CNN
    #        jj_model = load_model(model_dir + "jj_" + labeler_name)
    #        X = np.stack((j1_images_raw,j2_images_raw), axis = -1)
    #        X = standardize(*zero_center(X))[0]

    #    if(model_type == 4): #Dense
    #        jj_model = load_model(model_dir + "jj_" + labeler_name)
    #        X = np.append(j1_dense_inputs, j2_dense_inputs, axis = -1)
    #        print(X.shape)
    #        scores = jj_model.predict(X, batch_size = batch_size).reshape(-1)

    #    jj_scores = jj_model.predict(X, batch_size = batch_size).reshape(-1)
    #    jj_cut_vals = [np.percentile(jj_scores, thresh) for thresh in jj_threshholds]
    for i,thresh in enumerate(threshholds):
        print("Idx %i, thresh %.2f "  %(i, thresh))
        percentile = (100. - thresh)
        label = "X =%.0f%%" % percentile
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
        make_histogram(masses, labels, ['b','r'], 'Dijet Mass (GeV)', label, n_m_bins, 
                stacked = True, save = save_figs,  h_range = m_range, fname=plot_dir + options.label + "%.0fpcut_mass.png" %percentile)
        S = mjj[sig_events & in_window & pass_cut].shape[0]
        B  = max(mjj[bkg_events & in_window & pass_cut].shape[0], 1e-5)
        print("S = %.0f, B = %.0f, S/B %.3f, sigificance ~ %.1f " % (S, B, float(S)/ B, S/np.sqrt(B)))


    make_histogram(mjj_dists, dist_labels, colors_sculpt, 'Dijet Mass (GeV)', "", n_m_bins,
                   normalize=True, yaxis_label ="Arbitrary Units", save = save_figs,  h_range = m_range, fname=plot_dir + options.label + "qcd_mass_sculpt.png")

