import sys
sys.path.append('..')
from utils.TrainingUtils import *
from optparse import OptionParser
from optparse import OptionGroup

parser = OptionParser()
parser.add_option("-i", "--fin", default='../data/jet_images.h5', help="Input file for training.")
parser.add_option("--plot_dir", default='../plots/', help="Directory to output plots")
parser.add_option("--model_dir", default='../models/', help="Directory to read in and output models")
parser.add_option("--model_name", default='test.h5', help="Name of model to load")
parser.add_option("--model_type", type='int', default=0, help="0 CNN (one jet), 1 auto encoder, 2 dense (one jet), 3 CNN (both jets), 4 dense (both jets)")
parser.add_option("--plot_label", default="bumps_", help="what to call the plots")
parser.add_option("--num_data", type='int', default=200000)
parser.add_option("--data_start", type='int', default=600000)

(options, args) = parser.parse_args()

fin = options.fin
plot_dir = options.plot_dir
model_dir = options.model_dir



model_name = options.model_name
model_type = options.model_type
plot_label = options.plot_label

num_data = options.num_data
data_start = options.data_start

use_or = False
use_j = 0

sig_frac = -1.
signal = 1
sample_standardize = False




js_threshholds = [97., 95., 93., 90., 80., 70., 0.0]
jj_threshholds = [99.7, 99., 95., 90., 75., 0.0]


use_dense = (model_type == 2 or model_type == 4)

    


keys = ['j1_images', 'j2_images', 'jet_kinematics']
data = prepare_dataset(fin, signal_idx = signal, sig_frac = sig_frac, keys = keys, start = data_start, stop = data_start + num_data )
j1_images = np.expand_dims(data['j1_images'], axis = -1)
j2_images = np.expand_dims(data['j2_images'], axis = -1)
Y = data['label']
mjj = data['jet_kinematics'][:,0]
j1_m = data['jet_kinematics'][:,5]
j2_m = data['jet_kinematics'][:,9]

if(sample_standardize):
    j2_images = standardize(*zero_center(j2_images, np.zeros_like(j2_images)))[0]
    j1_images = standardize(*zero_center(j1_images, np.zeros_like(j1_images)))[0]


batch_size = 1000

save_figs = True
color = ['b']

if(model_type <= 2):
    threshholds = js_threshholds
    if(use_j != 0):
        threshholds = jj_threshholds
    j1_model = load_model(model_dir + "j1_" + model_name)
    j2_model = load_model(model_dir + "j2_" + model_name)
    if(model_type == 0):
        print("computing scores")
        j1_scores = j1_model.predict(j1_images, batch_size = batch_size)
        j2_scores = j2_model.predict(j2_images, batch_size = batch_size)
    elif(model_type ==1):
        j1_reco_images = j1_model.predict(j1_images, batch_size = batch_size)
        j2_reco_images = j2_model.predict(j2_images, batch_size = batch_size)
        j1_scores =  np.mean(keras.losses.mean_squared_error(j1_reco_images, j1_images), axis=(1,2)).reshape(-1)
        j2_scores =  np.mean(keras.losses.mean_squared_error(j2_reco_images, j2_images), axis=(1,2)).reshape(-1)
    elif(model_type == 2):
        j1_scores = j1_model.predict(j1_dense_inputs, batch_size = batch_size)
        j2_scores = j2_model.predict(j2_dense_inputs, batch_size = batch_size)

    j1_scores = j1_scores.reshape(-1)
    j2_scores = j2_scores.reshape(-1)
    j1_cut_vals = [np.percentile(j1_scores, thresh) for thresh in threshholds]
    j2_cut_vals = [np.percentile(j2_scores, thresh) for thresh in threshholds]
    #make_histogram(j1_scores, ['j1_scores'], color, 'J1 Score', '', 30, 
    #    stacked = True, save = save_figs,  fname=plot_dir + plot_label + "j1_scores.png")
    #make_histogram(j2_scores, ['j2_scores'], color, 'j2 Score', '', 30, 
    #    stacked = True, save = save_figs,  fname=plot_dir + plot_label + "j2_scores.png")
else:
    threshholds = jj_threshholds
    jj_model = load_model(model_dir + "jj_" + model_name)
    if(model_type == 3):
        X = np.stack((j1_images_raw,j2_images_raw), axis = -1)
        X = standardize(*zero_center(X))[0]
        jj_scores = jj_model.predict(X, batch_size = batch_size).reshape(-1)
    else:
        jj_scores = jj_model.predict(jj_dense_inputs, batch_size = batch_size).reshape(-1)
        
    jj_cut_vals = [np.percentile(jj_scores, thresh) for thresh in jj_threshholds]












n_m_bins = 30
m_range = (1200., 7000.)

mjj_dists = []

for i,thresh in enumerate(threshholds):
    print("Idx %i, thresh %.2f "  %(i, thresh))
    if(model_type <= 2):
        pass_cut = (j1_scores > j1_cut_vals[i]) & (j2_scores > j2_cut_vals[i])
        if(use_or): pass_cut = (j1_scores > j1_cut_vals[i]) | (j2_scores > j2_cut_vals[i])
        elif(use_j ==1):
            pass_cut = (j1_scores > j1_cut_vals[i])
        elif(use_j==2):
            pass_cut = (j2_scores > j2_cut_vals[i])

        j1_masses = [j1_m[ j1_scores > j1_cut_vals[i]]]
        j2_masses = [j2_m[ j2_scores > j2_cut_vals[i]]]
    else:
        pass_cut = (jj_scores > jj_cut_vals[i])
        j1_masses = [j1_m[ pass_cut]]
        j2_masses = [j2_m[ pass_cut]]

    masses = [mjj[pass_cut]]
    percentile = int(100. - thresh)
    make_histogram(masses, [''], color, 'Dijet Mass (GeV)', "Select top %i %%" % percentile, n_m_bins, 
            stacked = True, save = save_figs,  h_range = m_range, fname=plot_dir + plot_label + "%ipcut_mass.png" %percentile)

    #make_histogram(j1_masses, [''], color, 'Jet Mass (GeV)', "Select top %i %%" % percentile, n_m_bins, 
    #        stacked = True, save = save_figs, fname=plot_dir + plot_label + "%ipcut_j1_mass.png" %percentile)

    #make_histogram(j2_masses, [''], color, 'Jet Mass (GeV)', "Select top %i %%" % percentile, n_m_bins, 
    #        stacked = True, save = save_figs, fname=plot_dir + plot_label + "%ipcut_j2_mass.png" %percentile)



