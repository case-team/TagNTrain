import sys
sys.path.append('..')
from utils.TrainingUtils import *
import matplotlib.colors as mcolors
from optparse import OptionParser
from optparse import OptionGroup


parser = OptionParser()
parser.add_option("-i", "--fin", default='../data/jet_images.h5', help="Input file for training.")
parser.add_option("--plot_dir", default='../plots/BB_v2_M2500/', help="Directory to output plots")
parser.add_option("--model_dir", default='../models/BB_v2_M2500/', help="Directory to read in and output models")
parser.add_option("--model_name", default='', help="Name of model to load")
parser.add_option("--model_type", type='int', default=0, help="0 CNN (one jet), 1 auto encoder, 2 dense (one jet), 3 CNN (both jets), 4 dense (both jets)")
parser.add_option("-l", "--plot_label", default="", help="what to call the plots")
parser.add_option("--num_data", type='int', default=-1)
parser.add_option("--batch_start", type='int', default=-1, help="Train over multiple batches of dataset. Starting batch")
parser.add_option("--batch_stop", type='int', default=-1, help="Train over multiple batches of dataset. Stopping batch (inclusive)")
parser.add_option("--data_start", type='int', default=0)
parser.add_option("--mjj_low", type='int', default = -1,  help="Low mjj cut value")
parser.add_option("--mjj_high", type='int', default = -1, help="High mjj cut value")
parser.add_option("-s", "--signal", type='int', default=1, help="Which signal type to use ")
parser.add_option("--sig_frac",  type='float', default=-1., help="Filter signal to this amount (negative to do nothing)")
parser.add_option("--hadronic_only",  default=False, action='store_true',  help="Filter out leptonic decays")
parser.add_option("--d_eta", type='float', default = -1, help="Delta eta cut")


(options, args) = parser.parse_args()

fin = options.fin
plot_dir = options.plot_dir
model_dir = options.model_dir



model_name = options.model_name
model_type = options.model_type
plot_label = options.plot_label
if( len(plot_label) == 0):
    print("Must provide plot label (-l, --plot_label)")
    exit(1)
if(plot_label[-1] != '_'):
    plot_label += '_'

num_data = options.num_data
data_start = options.data_start

threshholds = [99.5, 99., 96., 90., 75., 0.0]
n_pts = len(threshholds)
color_threshholds = [100./n_pts * i for i in range(n_pts)]



keys = ["jet_kinematics", "j1_images", "j2_images", "jj_images", "j1_features", "j2_features", "jj_features"]

data = DataReader(fin, signal_idx = options.signal, sig_frac = 0., keys = keys, start = data_start, stop = data_start + num_data, hadronic_only = options.hadronic_only, 
        batch_start = options.batch_start, batch_stop = options.batch_stop, eta_cut = options.d_eta)
data.read()

mjj = data['jet_kinematics'][:,0]
j1_m = data['jet_kinematics'][:,5]
j2_m = data['jet_kinematics'][:,9]

j1_images = data['j1_images']
j2_images = data['j2_images']
jj_images = data['jj_images']
j1_dense_inputs = data['j1_features']
j2_dense_inputs = data['j2_features']
jj_dense_inputs = data['jj_features']
Y = data['label'].reshape(-1)


batch_size = 1000


if(model_type  <= 2 or model_type == 5): #classifier on each jet
    if(model_type <= 2):
        if('/' not in model_name):
            j1_fname = model_dir + "j1_" + model_name
            j2_fname = model_dir + "j2_" + model_name
        else:
            ins_idx = model_name.rfind('/')+1
            j1_fname = model_dir + model_name[:ins_idx] + "j1_" + model_name[ins_idx:]
            j2_fname = model_dir + model_name[:ins_idx] + "j2_" + model_name[ins_idx:]
        j1_model = tf.keras.models.load_model(j1_fname)
        j2_model = tf.keras.models.load_model(j2_fname)

        if(model_type == 0):  #CNN
            j1_score = j1_model.predict(j1_images, batch_size = batch_size)
            j2_score = j2_model.predict(j2_images, batch_size = batch_size)
        elif(model_type == 1): #autoencoder
            j1_reco_images = j1_model.predict(j1_images, batch_size=batch_size)
            j1_score =  np.mean(np.square(j1_reco_images - j1_images), axis=(1,2))
            j2_reco_images = j2_model.predict(j2_images, batch_size=batch_size)
            j2_score =  np.mean(np.square(j2_reco_images -  j2_images), axis=(1,2))
        elif(model_type == 2): #dense
            j1_score = j1_model.predict(j1_dense_inputs, batch_size = batch_size)
            j2_score = j2_model.predict(j2_dense_inputs, batch_size = batch_size)
    elif(model_type == 5): #VAE
        j1_model = VAE(0, model_dir = model_dir + "j1_" +  f)
        j1_model.load()
        j1_reco_images, j1_z_mean, j1_z_log_var = j1_model.predict_with_latent(j1_images)
        j1_score = compute_loss_of_prediction_mse_kl(j1_images, j1_reco_images, j1_z_mean, j1_z_log_var)[0]
        j2_model = VAE(0, model_dir = model_dir + "j2_" +  f)
        j2_model.load()
        j2_reco_images, j2_z_mean, j2_z_log_var = j2_model.predict_with_latent(j2_images)
        j2_score = compute_loss_of_prediction_mse_kl(j2_images, j2_reco_images, j2_z_mean, j2_z_log_var)[0]

    j1_score = j1_score.reshape(-1)
    j2_score = j2_score.reshape(-1)

elif(model_type == 3): #CNN both jets
    jj_model = tf.keras.models.load_model(model_dir + model_name)
    jj_score = jj_model.predict(jj_images, batch_size = batch_size).reshape(-1)

elif(model_type == 4): #Dense both jets
    jj_model = tf.keras.models.load_model(model_dir + model_name)
    jj_score = jj_model.predict(jj_dense_inputs, batch_size = batch_size).reshape(-1)










save_figs = True
labels = ['background', 'signal']
colors_sculpt = []
colormap = cm.viridis
normalize = mcolors.Normalize(vmin = 0., vmax=100.)



n_m_bins = 30
n_pt_bins = 30
m_range = (1500., 5000.)
pt_range = (500., 4000.)

#j1_pt = j1_4vec[:,0]
#j2_pt = j2_4vec[:,0]


sig_events = (Y > 0.9)
bkg_events = (Y < 0.1)
mjj_dists = []
j1_pt_dists = []
j2_pt_dists = []
dist_labels = []
#mjj_dists.append(mjj[bkg_events])

for i,thresh in enumerate(threshholds):
    print("Idx %i, thresh %.2f "  %(i, thresh))
    percentile = (100. - threshholds[i])
    if(thresh == 0.): label = "No Selection"
    else: label = "Eff = %.1f%%" % percentile
    dist_labels.append(label)
    if(model_type <= 2):
        pass_cut = make_selection(j1_score, j2_score, threshholds[i])
    else:
        thresh = np.percentile(jj_score, threshholds[i])
        pass_cut =  jj_score > thresh

    colors_sculpt.append(colormap(normalize(color_threshholds[i])))
    sig_pass_cut = pass_cut & sig_events
    bkg_pass_cut = pass_cut & bkg_events
    masses = [mjj[bkg_pass_cut], mjj[sig_pass_cut]]
    mjj_dists.append(mjj[bkg_pass_cut])
    #j1_pt_dists.append(j1_pt[bkg_pass_cut])
    #j2_pt_dists.append(j2_pt[bkg_pass_cut])
    #FIXME Turn into histogram first
    #js_div = JS_Distance(mjj_dists[0], mjj[bkg_events & pass_cut])
    #print("Mean mass, JS divergence is : ", np.mean(mjj_dists[i]), js_div)
    #make_histogram(masses, labels, ['b','r'], 'Dijet Mass (GeV)', label, n_m_bins, 
            #stacked = True, save = save_figs,  h_range = m_range, fname=plot_dir + plot_label + "%.0fpcut_mass.png" %percentile)

#j1_pt_dists = [j1_pt[bkg_events & (j1_scores > j1_cut_vals[i]) & (j2_scores > j2_cut_vals[i])] for i in range(len(threshholds))] 
#j2_pt_dists = [j2_pt[bkg_events & (j1_scores > j1_cut_vals[i]) & (j2_scores > j2_cut_vals[i])] for i in range(len(threshholds))] 

make_histogram(mjj_dists, dist_labels, colors_sculpt, 'Dijet Mass (GeV)', "", n_m_bins,
               normalize=True, yaxis_label ="Arbitrary Units", save = save_figs,  h_range = m_range, fname=options.plot_dir + plot_label + "qcd_mass_sculpt.png")

#make_histogram(j1_pt_dists, dist_labels, colors_sculpt, 'J1 Pt(GeV)', "QCD J1 Pt distribution", n_pt_bins,
#               normalize=True, save = save_figs,  h_range = pt_range, fname=plot_dir + plot_label + "qcd_j1_pt_sculpt.png")
#make_histogram(j2_pt_dists, dist_labels, colors_sculpt, 'J2 Pt(GeV)', "QCD J2 Pt distribution", n_pt_bins,
#               normalize=True, save = save_figs,  h_range = pt_range, fname=plot_dir + plot_label + "qcd_j2_pt_sculpt.png")
del data
