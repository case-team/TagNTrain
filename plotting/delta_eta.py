import sys
sys.path.append('..')
from utils.TrainingUtils import *
import matplotlib.colors as mcolors
from optparse import OptionParser
from optparse import OptionGroup

parser = input_options()
parser.add_argument("--plot_label", default='', help="extra str for plot label")
options = parser.parse_args()

eta_cut = options.d_eta
options.d_eta = -1.
options.keys = ['jet_kinematics']

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





    

data, _ = load_dataset_from_options(options)
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


eff_B = np.sum(d_eta[bkg_events] < eta_cut) / d_eta[bkg_events].shape[0]
eff_S = np.sum(d_eta[sig_events] < eta_cut) / d_eta[sig_events].shape[0]

print("Testing eta cut %.1f. Eff sig %.2f Eff bkg %.2f. Significance improvement %.2f \n" %  (eta_cut, eff_S, eff_B, eff_S / np.sqrt(eff_B)));
