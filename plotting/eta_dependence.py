import sys
sys.path.append('..')
from utils.TrainingUtils import *
import matplotlib.colors as mcolors
from optparse import OptionParser
from optparse import OptionGroup

parser = OptionParser()
parser.add_option("-i", "--fin", default='../data/jet_images.h5', help="Input file for training.")
parser.add_option("--plot_dir", default='../plots/', help="Directory to output plots")
parser.add_option("--model_dir", default='../models/', help="Directory to read in and output models")
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
parser.add_option("--mjj_sig", type='int', default = -1, help="Mjj value of signal (used for filtering in correct mass region)")
parser.add_option("--hadronic_only",  default=False, action='store_true',  help="Filter out leptonic decays")


(options, args) = parser.parse_args()

if(options.plot_dir[-1] != '/'): options.plot_dir+="/"
if(options.plot_label[-1] != '_'): options.plot_label+= "_"

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

use_or = False
use_j = 0

sample_standardize = False

mjj_window = 500.




use_dense = (model_type == 2 or model_type == 4)

    

keys = ['jet_kinematics']
if(options.model_name != "" and not use_dense):
    keys += ['j1_images', 'j2_images']
elif(options.model_name != "" and use_dense): 
    keys += ["j1_features", "j2_features"]

data = DataReader(fin, signal_idx = options.signal, sig_frac = options.sig_frac, keys = keys, start = data_start, stop = data_start + num_data, hadronic_only = options.hadronic_only, 
        batch_start = options.batch_start, batch_stop = options.batch_stop, m_sig = options.mjj_sig, m_low = options.mjj_low, m_high = options.mjj_high)
data.read()
Y = data['label'].reshape(-1)
mjj = data['jet_kinematics'][:,0]
deta = data['jet_kinematics'][:,1]
j1_m = data['jet_kinematics'][:,5]
j2_m = data['jet_kinematics'][:,9]
j1_images = None
j2_images = None
j1_dense_inputs = None
j2_dense_inputs = None

if(options.model_name != "" and not use_dense):
    j1_images = data['j1_images']
    j2_images = data['j2_images']
elif(options.model_name != "" and use_dense): 
    j1_dense_inputs = data['j1_features']
    j2_dense_inputs = data['j2_features']


batch_size = 1000





save_figs = True
labels = ['background', 'signal']
#colors = ['b', 'r', 'g', 'purple', 'pink', 'black', 'magenta', 'pink']
colors = ["g", "gray", "b", "r","m", "skyblue", "pink"]
colors_sculpt = []
colormap = cm.viridis
normalize = mcolors.Normalize(vmin = 0., vmax=100.)




sig_events = (Y > 0.9)
bkg_events = (Y < 0.1)

eta_bins = np.linspace(0.,2.5,25)

if(options.model_name == ""): #apply selection
    print("Must provide model with --model_name option!")
    sys.exit(1)

else:
    if(model_type <= 2):
        j1_scores, j2_scores = get_jet_scores(options.model_dir, options.model_name, options.model_type, j1_images, j2_images, j1_dense_inputs, j2_dense_inputs)

        make_profile_hist(deta[bkg_events], j1_scores[bkg_events], eta_bins, xaxis_label = r'$|\Delta\eta|$', yaxis_label = "J1 Score",  
                fname = options.plot_dir + options.plot_label + "j1_eta_dependence.png")
        make_profile_hist(deta[bkg_events], j2_scores[bkg_events], eta_bins, xaxis_label = r'$|\Delta\eta|$', yaxis_label = "J2 Score",  
                fname = options.plot_dir + options.plot_label + "j2_eta_dependence.png")

    else:
        if(model_type == 3): #CNN
            jj_model = load_model(model_dir + "jj_" + model_name)
            X = np.stack((j1_images_raw,j2_images_raw), axis = -1)
            X = standardize(*zero_center(X))[0]

        if(model_type == 4): #Dense
            jj_model = load_model(model_dir + "jj_" + model_name)
            X = np.append(j1_dense_inputs, j2_dense_inputs, axis = -1)
            print(X.shape)
            scores = jj_model.predict(X, batch_size = batch_size).reshape(-1)

        jj_scores = jj_model.predict(X, batch_size = batch_size).reshape(-1)


