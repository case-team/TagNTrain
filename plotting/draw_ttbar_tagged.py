import sys
import subprocess
sys.path.append('..')
from utils.TrainingUtils import *
import matplotlib.colors as mcolors
from optparse import OptionParser
from optparse import OptionGroup

parser = input_options()
parser.add_argument("--effs", nargs="+", default = [], type = float)
options = parser.parse_args()

if(not os.path.exists(options.output)):
    subprocess.call("mkdir %s" % options.output, shell = True)

fin = options.fin
plot_dir = options.plot_dir
model_dir = options.model_dir



    

options.keys = ['j1_features', 'j2_features', 'jet_kinematics']
if(len(options.sig_file) == 0): 
    options.sig_per_batch = 0
#options.keep_mlow = 1300.
#options.keep_mhigh = 1800.
#options.keep_mhigh = 99999.
options.ptsort = True


data, _ = load_dataset_from_options(options)

Y = data['label'].reshape(-1)
mjj = data['jet_kinematics'][:,0]
j1_pt = data['jet_kinematics'][:,2]
j2_pt = data['jet_kinematics'][:,6]

j1_m = data['j1_features'][:,0]
j2_m = data['j2_features'][:,0]

j1_tau21 = data['j1_features'][:,2]
j1_tau32 = data['j1_features'][:,3]
j1_deepcsv = data['j1_features'][:,6]

j2_tau21 = data['j2_features'][:,2]
j2_tau32 = data['j2_features'][:,3]
j2_deepcsv = data['j2_features'][:,6]

#QCD: 0 Single Top: -1, ttbar: -2, V+jets: -3

sig = (Y == 1)
QCD = (Y == 0)
ttbar = (Y == -2)
other = (Y == -1) | (Y == -3)

pt_cut = (j1_pt > 400.) & (j2_pt > 400.)

#proper cuts are UL16preVFP 0.2027, UL16PostVFP 0.1918, UL17 0.1355, UL18 0.1208
#https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
# use an average
deepcsv_cut = 0.16
tau32_cut = 0.75

#lower pt side is used to 'tag'
pre_tag_selection = pt_cut & (j2_tau32 < tau32_cut) & (j2_deepcsv > deepcsv_cut)


probe_scores = get_single_jet_scores(options.labeler_name, options.model_type, j_images = None, j_dense_inputs = data['j1_features'][pre_tag_selection], 
        num_models = options.num_models, batch_size = 512)

percentile_cut = 100. - options.effs[0]
thresh = np.percentile(probe_scores, percentile_cut)

nn_cut = np.copy(pre_tag_selection)
nn_cut[pre_tag_selection] = probe_scores > thresh



pass_cut = pre_tag_selection & nn_cut

print(np.mean(pre_tag_selection))
print(np.mean(pass_cut))
print(np.mean(probe_scores > thresh))


ttbar_frac_before = j2_m[pre_tag_selection & ttbar].shape[0] / j2_m[pre_tag_selection].shape[0]
ttbar_frac_after = j2_m[pass_cut & ttbar].shape[0] / j2_m[pass_cut].shape[0]


sig_frac_before = j2_m[pre_tag_selection & sig].shape[0] / j2_m[pre_tag_selection].shape[0]
sig_frac_after = j2_m[pass_cut & sig].shape[0] / j2_m[pass_cut].shape[0]

print("Before: %i events, %.3f ttbar frac %.3f sig frac" % (j2_m[pre_tag_selection].shape[0], ttbar_frac_before, sig_frac_before))
print("After tag: %i events, %.3f ttbar frac %.3f sig frac" % (j2_m[pass_cut].shape[0], ttbar_frac_after, sig_frac_before))


save_figs = True
labels = ['QCD', 'ttbar', 'tW and V+jets']
colors = ["b", "r", "g"] # "m", "skyblue", "pink"]

masks = [QCD, ttbar, other]

if(sig_frac_before > 0):
    labels.append("Signal")
    colors.append("m")
    masks.append(sig)



n_m_bins = 9
m_range = (105, 285)
n_feat_bins = 10
feat_range = (0., 1.)

def hist_list(var, cut, masks):
    return [var[cut & mask] for mask in masks]

make_histogram(hist_list(j1_m, pt_cut, masks), labels, colors, xaxis_label = 'Jet Mass', title = "Pre-Selection", 
    num_bins = n_m_bins, h_range = m_range, stacked = True, fname = options.output + "jet_mass_presel_tag.png")
make_histogram(hist_list(j2_m, pt_cut, masks), labels, colors, xaxis_label = 'Jet Mass', title = " Pre-Selection", 
    num_bins = n_m_bins, h_range = m_range, stacked = True, fname = options.output + "jet_mass_presel_probe.png")

make_histogram(hist_list(j2_m, pass_cut, masks), labels, colors, xaxis_label = 'Jet Mass', title = " After J2 Top-Tag", 
    num_bins = n_m_bins, h_range = m_range, stacked = True, fname = options.output + "jet_mass_pass_tag.png")

make_histogram(hist_list(j2_m, pre_tag_selection, masks), labels, colors, xaxis_label = 'Jet Mass', title = " 'After J1 Cuts' ", 
    num_bins = n_m_bins, h_range = m_range, stacked = True, fname = options.output + "jet_mass_before_tag.png")

make_histogram(hist_list(j1_m, pass_cut, masks), labels, colors, xaxis_label = 'Jet Mass', title = " After J2 Top-Tag", 
    num_bins = n_m_bins, h_range = m_range, stacked = True, fname = options.output + "jet_mass_pass_probe.png")

make_histogram(hist_list(j1_m, pre_tag_selection, masks), labels, colors, xaxis_label = 'Jet Mass', title = " 'After J1 Cuts' ", 
    num_bins = n_m_bins, h_range = m_range, stacked = True, fname = options.output + "jet_mass_before_probe.png")


