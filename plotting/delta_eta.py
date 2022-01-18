import sys
sys.path.append('..')
from utils.TrainingUtils import *
import matplotlib.colors as mcolors
from optparse import OptionParser
from optparse import OptionGroup

parser = input_options()
parser.add_argument("--do_ttbar", default=False, action = 'store_true',  help="Draw ttbar")
options = parser.parse_args()

eta_cut = options.deta
options.deta = -1.
options.keys = ['jet_kinematics']

fin = options.fin



plot_label = options.label
if( len(plot_label) == 0):
    print("Must provide plot label (-l, --plot_label)")
    exit(1)
if(plot_label[-1] != '_'):
    plot_label += '_'

if(options.do_ttbar):
    options.sig_per_batch = 0




    

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
eta_range = (0, 5.0)



sig_events = (Y > 0.9)
bkg_events = (Y < 0.1)

if(options.do_ttbar):
    bkg_events = Y == 0
    sig_events = Y == -2

mjj_dists = []
j1_pt_dists = []
j2_pt_dists = []
dist_labels = []
#mjj_dists.append(mjj[bkg_events])

#No selection

title = "No Selection"
if(options.keep_mlow > 0 and options.keep_mhigh > 0):
    title = "%.0f > Mjj > Mjj %.0f" % (options.keep_mlow, options.keep_mhigh)
make_outline_hist([d_eta[bkg_events]],  d_eta[sig_events], labels, ['b','r'], r'Dijet $|\Delta\eta|$', title, n_bins, logy = True,
        normalize = True, save = save_figs,  h_range = eta_range, fname=options.output + plot_label + "nocut_delta_eta.png" )


if(eta_cut > 0):
    eff_B = np.sum(d_eta[bkg_events] < eta_cut) / d_eta[bkg_events].shape[0]
    eff_S = np.sum(d_eta[sig_events] < eta_cut) / d_eta[sig_events].shape[0]

    print("Before Cut S = %.1f, B = %.1f" % ( d_eta[sig_events].shape[0], d_eta[bkg_events].shape[0]))


    print("Testing eta cut %.1f. Eff sig %.2f Eff bkg %.2f. Significance improvement %.2f \n" %  (eta_cut, eff_S, eff_B, eff_S / np.sqrt(eff_B)));
