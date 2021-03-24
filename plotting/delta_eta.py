import sys
sys.path.append('..')
from utils.TrainingUtils import *
import matplotlib.colors as mcolors
from optparse import OptionParser
from optparse import OptionGroup

parser = OptionParser()
parser.add_option("-i", "--fin", default='../data/jet_images.h5', help="Input file for training.")
parser.add_option("--plot_dir", default='../plots/', help="Directory to output plots")
parser.add_option("-l", "--plot_label", default="", help="what to call the plots")
parser.add_option("--num_data", type='int', default=-1)
parser.add_option("--batch_start", type='int', default=-1, help="Train over multiple batches of dataset. Starting batch")
parser.add_option("--batch_stop", type='int', default=-1, help="Train over multiple batches of dataset. Stopping batch (inclusive)")
parser.add_option("--data_start", type='int', default=0)
parser.add_option("-s", "--signal", type='int', default=1, help="Which signal type to use ")
parser.add_option("--sig_frac",  type='float', default=-1., help="Filter signal to this amount (negative to do nothing)")
parser.add_option("--mjj_sig", type='int', default = -1, help="Mjj value of signal (used for filtering in correct mass region)")
parser.add_option("--mjj_low", type='int', default = -1,  help="Low mjj cut value")
parser.add_option("--mjj_high", type='int', default = -1, help="High mjj cut value")
parser.add_option("--hadronic_only",  default=False, action='store_true',  help="Filter out leptonic decays")


(options, args) = parser.parse_args()

fin = options.fin
plot_dir = options.plot_dir



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


mjj_window = 500.





    

keys = ['jet_kinematics']
data = DataReader(fin, signal_idx = options.signal, sig_frac = options.sig_frac, keys = keys, start = data_start, stop = data_start + num_data, hadronic_only = options.hadronic_only, 
        batch_start = options.batch_start, batch_stop = options.batch_stop, m_low = options.mjj_low, m_high = options.mjj_high, m_sig = options.mjj_sig)
data.read()
Y = data['label'].reshape(-1)
mjj = data['jet_kinematics'][:,0]
d_eta = data['jet_kinematics'][:,1]






save_figs = True
labels = ['background', 'signal']
#colors = ['b', 'r', 'g', 'purple', 'pink', 'black', 'magenta', 'pink']
colors = ["g", "gray", "b", "r","m", "skyblue", "pink"]
colors_sculpt = []
colormap = cm.viridis



n_bins = 25
eta_range = (0, 2.5)



sig_events = (Y > 0.9)
bkg_events = (Y < 0.1)
mjj_dists = []
j1_pt_dists = []
j2_pt_dists = []
dist_labels = []
#mjj_dists.append(mjj[bkg_events])

#No selection

title = "No Selection"
if(options.mjj_low > 0 and options.mjj_high > 0):
    title = "%.0f > Mjj > Mjj %.0f" % (options.mjj_low, options.mjj_high)
make_outline_hist([d_eta[bkg_events]],  d_eta[sig_events], labels, ['b','r'], r'Dijet $|\Delta\eta|$', title, n_bins, logy = True,
        normalize = True, save = save_figs,  h_range = eta_range, fname=plot_dir + plot_label + "nocut_delta_eta.png" )


eta_cut = 1.1
eff_B = np.sum(d_eta[bkg_events] < eta_cut) / d_eta[bkg_events].shape[0]
eff_S = np.sum(d_eta[sig_events] < eta_cut) / d_eta[sig_events].shape[0]

print("Testing eta cut %.1f. Eff sig %.2f Eff bkg %.2f. Significance improvement %.2f \n" %  (eta_cut, eff_S, eff_B, eff_S / np.sqrt(eff_B)));
