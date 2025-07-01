import sys
import subprocess
sys.path.append('..')
from utils.TrainingUtils import *
import matplotlib.colors as mcolors
from optparse import OptionParser
from optparse import OptionGroup

def hist_list(var, cut, masks):
    return [var[cut & mask] for mask in masks]

parser = input_options()
parser.add_argument("--data_vals", default ="", help = "File with data masses")
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
options.ptsort = False
options.randsort = True


data, _ = load_dataset_from_options(options)

Y = data['label'].reshape(-1)
mjj = data['jet_kinematics'][:,0]
deta = data['jet_kinematics'][:,1]
j1_pt = data['jet_kinematics'][:,2]
j2_pt = data['jet_kinematics'][:,6]

j1_m = data['j1_features'][:,0]
j2_m = data['j2_features'][:,0]

data_masses = None
if(len(options.data_vals) > 0):
    with h5py.File(options.data_vals) as f:
        data_masses = f['mass'][:]

#print(j1_pt[:10])
#print(j2_pt[:10])
#print(j1_m[:10])
#print(j2_m[:10])

j1_tau21 = data['j1_features'][:,2]
j1_tau32 = data['j1_features'][:,3]
j1_deepcsv = data['j1_features'][:,-2]

j2_tau21 = data['j2_features'][:,2]
j2_tau32 = data['j2_features'][:,3]
j2_deepcsv = data['j2_features'][:,-2]

#QCD: 0 Single Top: -1, ttbar: -2, V+jets: -3

signal = (Y == 1)
QCD = (Y == 0)
ttbar = (Y == -2)
other = (Y == -1) | (Y == -3)

labels = ['QCD', 'tW and V+jets', 'top']
colors = [c_lightblue, c_orange, c_red]
masks = [QCD, other, ttbar]


plot_hist_stack( hist_list(mjj, mjj > 0., masks), labels, colors, xlabel = 'Mjj', title = "", 
    nbins = 30, h_range = (1200, 5000), logy = True, fname = options.output + "Mjj_preselection.png" )

plot_hist_stack(hist_list(deta, mjj > 0., masks), labels, colors, xlabel = r'Dijet $|\Delta \eta|$', title = "", 
    nbins = 30, h_range = (0., 3.), fname = options.output + "deta_preselection.png" )

print("Total number of signal events: " , np.sum(signal))

pt_cut = (j1_pt > 400.) & (j2_pt > 400.)

#proper cuts are UL16preVFP 0.2027, UL16PostVFP 0.1918, UL17 0.1355, UL18 0.1208
#https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
# use an average
#deepcsv_cut = 0.15
deepcsv_cut = -999.

#lower pt side is used to 'tag'
#tag_tau32 = 0.75
tag_tau32 = 999.
tag_selection = (j2_tau32 < tag_tau32) & (j2_deepcsv > deepcsv_cut)
if(options.deta < 0.): options.deta = 1.3

tag_selection = tag_selection & (deta < options.deta)


mlow = 150
mhigh = 250
full_tag_selection = tag_selection & (j2_m > mlow) & (j2_m < mhigh)
sideband_tag_selection = tag_selection & ((j2_m < mlow) | (j2_m > mhigh))

SR_cut = pt_cut & full_tag_selection 
SB_cut = pt_cut & sideband_tag_selection

print(np.mean(SB_cut))
print(np.mean(SR_cut))
print(np.mean(tag_selection))


print("%i sig events after tag selection"  % (np.sum(Y[tag_selection & pt_cut] == 1)))

print("SR")
print(j1_m[SR_cut])
print(j2_m[SR_cut])
print(Y[SR_cut])


#print("SB")
#print(j1_m[SB_cut])
#print(j2_m[SB_cut])
#print(Y[SB_cut])


ttbar_frac_SR = j2_m[SR_cut & ttbar].shape[0] / j2_m[SR_cut].shape[0]
ttbar_signif_SR = j2_m[SR_cut & ttbar].shape[0] / j2_m[SR_cut].shape[0]**0.5

ttbar_frac_SB = j2_m[SB_cut & ttbar].shape[0] / j2_m[SB_cut].shape[0]
ttbar_signif_SB = j2_m[SB_cut & ttbar].shape[0] / j2_m[SB_cut].shape[0]**0.5


sig_frac_SR = j2_m[SR_cut & signal].shape[0] / j2_m[SR_cut].shape[0]
sig_frac_SB = j2_m[SB_cut & signal].shape[0] / j2_m[SB_cut].shape[0]



print("SR: %i events, %.3f ttbar frac, signif %.2f" % (j2_m[SR_cut].shape[0], ttbar_frac_SR, ttbar_signif_SR))
print("SB: %i events, %.3f ttbar frac, signif %.2f" % (j2_m[SB_cut].shape[0], ttbar_frac_SB, ttbar_signif_SB))




save_figs = True

if(sig_frac_SR > 0):
    labels.append("Signal")
    colors.append("m")
    masks.append(signal)


n_m_bins = 9
m_range = (30, 285)
n_feat_bins = 10
feat_range = (0., 1.)


make_histogram(data_masses, ["data"], ["blue"], xaxis_label = 'Jet Mass', title = "", 
    num_bins = n_m_bins, h_range = m_range, stacked = False, fname = options.output + "data_masses.png")


plot_hist_stack(hist_list(j1_m, tag_selection, masks), labels, colors, xlabel = 'J1 $m_{SD}$ [GeV]', title = options.label, data_vals = data_masses,
    nbins = n_m_bins, h_range = m_range, fname = options.output + "j1_mass_stack_preselection.pdf")

plot_hist_stack(hist_list(j2_m, tag_selection, masks), labels, colors, xlabel = 'J2 $m_{SD}$ [GeV]', title = options.label, data_vals = data_masses,
    nbins = n_m_bins, h_range = m_range, fname = options.output + "j2_mass_stack_preselection.pdf")

make_histogram(hist_list(j1_m, tag_selection, masks), labels, colors, normalize = True, xaxis_label = 'Jet Mass', title = "", 
    num_bins = n_m_bins, h_range = m_range, stacked = True, fname = options.output + "jet_mass_preselection.png")

make_histogram(hist_list(j1_m, tag_selection, masks), labels, colors, normalize = True, xaxis_label = 'Jet Mass', title = "", 
    logy = True, num_bins = n_m_bins, h_range = m_range, stacked = True, fname = options.output + "jet_mass_preselection_logy.png")


make_histogram(hist_list(j1_m, SR_cut, masks), labels, colors, xaxis_label = 'Jet Mass', title = " Signal-Region ", 
    num_bins = n_m_bins, h_range = m_range, stacked = True, fname = options.output + "jet_mass_SR_probe.png")

make_histogram(hist_list(j1_m, SB_cut, masks), labels, colors, xaxis_label = 'Jet Mass', title = " Background-Region ", 
    num_bins = n_m_bins, h_range = m_range, stacked = True, fname = options.output + "jet_mass_SB_probe.png")


make_histogram(hist_list(j1_tau21, SR_cut, masks), labels, colors, xaxis_label = 'Tau21', title = " Signal-Region ", logy = True,
    num_bins = n_feat_bins, h_range = feat_range, stacked = True, fname = options.output + "tau21_SR_probe.png")


make_histogram(hist_list(j1_tau32, SR_cut, masks), labels, colors, xaxis_label = 'Tau32', title = " Signal-Region ",  logy = True,
    num_bins = n_feat_bins, h_range = feat_range, stacked = True, fname = options.output + "tau32_SR_probe.png")

make_histogram(hist_list(j1_deepcsv, SR_cut, masks), labels, colors, xaxis_label = 'DeepCSV', title = " Signal-Region ",  logy = True,
    num_bins = n_feat_bins, h_range = feat_range, stacked = True, fname = options.output + "deepcsv_SR_probe.png")


make_histogram(hist_list(j1_tau21, SB_cut, masks), labels, colors, xaxis_label = 'Tau21', title = " Background-Region ",  logy = True,
    num_bins = n_feat_bins, h_range = feat_range, stacked = True, fname = options.output + "tau21_SB_probe.png")

make_histogram(hist_list(j1_tau32, SB_cut, masks), labels, colors, xaxis_label = 'Tau32', title = " Background-Region ",  logy = True,
    num_bins = n_feat_bins, h_range = feat_range, stacked = True, fname = options.output + "tau32_SB_probe.png")

make_histogram(hist_list(j1_deepcsv, SB_cut, masks), labels, colors, xaxis_label = 'DeepCSV', title = " Background-Region ",  logy = True,
    num_bins = n_feat_bins, h_range = feat_range, stacked = True, fname = options.output + "deepcsv_SB_probe.png")
