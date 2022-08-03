import sys
sys.path.append('..')
from utils.TrainingUtils import *
import matplotlib.colors as mcolors
from optparse import OptionParser
from optparse import OptionGroup


parser = input_options()
parser.add_argument("--use_SB", default=False, action = 'store_true',  help="Define efficiency in SB's")
options = parser.parse_args()







#if( len(plot_label) == 0):
#    print("Must provide plot label (-l, --plot_label)")
#    exit(1)
#if(plot_label[-1] != '_'):
#    plot_label += '_'

num_data = options.num_data
data_start = options.data_start

threshholds = [99.5, 99., 96., 90., 75., 0.0]
n_pts = len(threshholds)
color_threshholds = [100./n_pts * i for i in range(n_pts)]



options.keys = ["jet_kinematics",  "j1_features", "j2_features"]

data, _ = load_dataset_from_options(options)

mjj = data['jet_kinematics'][:,0]
j1_m = data['jet_kinematics'][:,5]
j2_m = data['jet_kinematics'][:,9]


j1_images = None
j2_images = None
jj_images = None

#j1_images = data['j1_images']
#j2_images = data['j2_images']
#jj_images = data['jj_images']
j1_dense_inputs = data['j1_features']
j2_dense_inputs = data['j2_features']
#jj_dense_inputs = data['jj_features']
jj_dense_inputs = None
Y = data['label'].reshape(-1)


batch_size = 1000

if(options.use_SB):
    compute_mjj_window(options)
    print("%.0f %.0f %.0f %.0f" % (options.keep_mlow, options.mjj_low, options.mjj_high, options.keep_mhigh))
    in_SB = (((mjj > options.keep_mlow) & (mjj < options.mjj_low))  | (mjj > options.mjj_high) & (mjj < options.keep_mhigh))
    print("SB frac is %.2f" % np.mean(in_SB))
else: in_SB = mjj > 0.


print("Using model %s" % options.labeler_name)
if(options.model_type <= 2 or options.model_type == 5): #classifier on each jet

    j1_score, j2_score = get_jet_scores("", options.labeler_name, options.model_type, j1_images, j2_images, j1_dense_inputs, j2_dense_inputs, num_models = options.num_models)
    j1_QT = QuantileTransformer(copy = True)
    j1_qs = j1_QT.fit_transform(j1_score.reshape(-1,1)).reshape(-1)
    j2_QT = QuantileTransformer(copy = True)
    j2_qs = j2_QT.fit_transform(j2_score.reshape(-1,1)).reshape(-1)

    jj_scores = j1_qs * j2_qs
else:
    jj_scores = get_jj_scores("", f, options.model_type, jj_images, jj_dense_inputs, num_models = options.num_models)
        











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

    thresh = np.percentile(jj_scores[in_SB], threshholds[i])
    print("Perc %.2f, thresh %.4f " % (percentile, thresh))
    pass_cut =  jj_scores > thresh

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

make_histogram(mjj_dists, dist_labels, colors_sculpt, 'Dijet Mass (GeV)', "", n_m_bins, logy = True,
               normalize=True, yaxis_label ="Arbitrary Units",  h_range = m_range, fname=options.output + options.label + "qcd_mass_sculpt.png")

#make_histogram(j1_pt_dists, dist_labels, colors_sculpt, 'J1 Pt(GeV)', "QCD J1 Pt distribution", n_pt_bins,
#               normalize=True, save = save_figs,  h_range = pt_range, fname=plot_dir + plot_label + "qcd_j1_pt_sculpt.png")
#make_histogram(j2_pt_dists, dist_labels, colors_sculpt, 'J2 Pt(GeV)', "QCD J2 Pt distribution", n_pt_bins,
#               normalize=True, save = save_figs,  h_range = pt_range, fname=plot_dir + plot_label + "qcd_j2_pt_sculpt.png")
del data
