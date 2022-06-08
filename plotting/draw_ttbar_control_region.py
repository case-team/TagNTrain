import sys
import subprocess
sys.path.append('..')
from utils.TrainingUtils import *
import matplotlib.colors as mcolors
from optparse import OptionParser
from optparse import OptionGroup

parser = input_options()
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

#print(j1_pt[:10])
#print(j2_pt[:10])
#print(j1_m[:10])
#print(j2_m[:10])

j1_tau21 = data['j1_features'][:,2]
j1_tau32 = data['j1_features'][:,3]
j1_deepcsv = data['j1_features'][:,6]

j2_tau21 = data['j2_features'][:,2]
j2_tau32 = data['j2_features'][:,3]
j2_deepcsv = data['j2_features'][:,6]

#QCD: 0 Single Top: -1, ttbar: -2, V+jets: -3

signal = (Y == 1)
QCD = (Y == 0)
ttbar = (Y == -2)
other = (Y == -1) | (Y == -3)

print("Total number of signal events: " , np.sum(signal))

pt_cut = (j1_pt > 400.) & (j2_pt > 400.)

#proper cuts are UL16preVFP 0.2027, UL16PostVFP 0.1918, UL17 0.1355, UL18 0.1208
#https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
# use an average
deepcsv_cut = 0.16
#deepcsv_cut = -999.

#lower pt side is used to 'tag'
tag_tau32 = 0.75
#tag_tau32 = 999.
tag_selection = (j2_tau32 < tag_tau32) & (j2_deepcsv > deepcsv_cut)
if(options.deta_min > 0): 
    tag_selection = tag_selection & (data['jet_kinematics'][:,1] > options.deta_min)


full_tag_selection = tag_selection & (j2_m > 105) & (j2_m < 220)
sideband_tag_selection = tag_selection & ((j2_m < 105) | (j2_m > 220))
probe_selection = (j1_tau32 < 0.65) & (j1_deepcsv > deepcsv_cut)

pass_cut = pt_cut & full_tag_selection & probe_selection
fail_cut = pt_cut & full_tag_selection & (~probe_selection)

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
ttbar_frac_SB = j2_m[SB_cut & ttbar].shape[0] / j2_m[SB_cut].shape[0]


sig_frac_SR = j2_m[SR_cut & signal].shape[0] / j2_m[SR_cut].shape[0]
sig_frac_SB = j2_m[SB_cut & signal].shape[0] / j2_m[SB_cut].shape[0]



print("SR: %i events, %.3f ttbar frac, %.5f sig frac" % (j2_m[SR_cut].shape[0], ttbar_frac_SR, sig_frac_SR))
print("SB: %i events, %.3f ttbar frac, %.5f sig frac" % (j2_m[SB_cut].shape[0], ttbar_frac_SB, sig_frac_SB))




save_figs = True
labels = ['QCD', 'ttbar', 'tW and V+jets']
colors = ["b", "r", "g"] # "m", "skyblue", "pink"]
masks = [QCD, ttbar, other]

if(sig_frac_SR > 0):
    labels.append("Signal")
    colors.append("m")
    masks.append(signal)


n_m_bins = 9
m_range = (0, 285)
n_feat_bins = 10
feat_range = (0., 1.)

def hist_list(var, cut, masks):
    return [var[cut & mask] for mask in masks]

make_histogram(hist_list(j2_m, pass_cut, masks), labels, colors, xaxis_label = 'Jet Mass', title = " 'Pass' ", 
    num_bins = n_m_bins, h_range = m_range, stacked = True, fname = options.output + "jet_mass_pass_tag.png")

make_histogram(hist_list(j2_m, fail_cut, masks), labels, colors, xaxis_label = 'Jet Mass', title = " 'Fail' ", 
    num_bins = n_m_bins, h_range = m_range, stacked = True, fname = options.output + "jet_mass_fail_tag.png")

make_histogram(hist_list(j1_m, pass_cut, masks), labels, colors, xaxis_label = 'Jet Mass', title = " 'Pass' ", 
    num_bins = n_m_bins, h_range = m_range, stacked = True, fname = options.output + "jet_mass_pass_probe.png")

make_histogram(hist_list(j1_m, fail_cut, masks), labels, colors, xaxis_label = 'Jet Mass', title = " 'Fail' ", 
    num_bins = n_m_bins, h_range = m_range, stacked = True, fname = options.output + "jet_mass_fail_probe.png")


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
