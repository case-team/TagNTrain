import sys
sys.path.append('..')
from utils.TrainingUtils import *
from optparse import OptionParser
from optparse import OptionGroup




parser = input_options()
parser.add_argument("--do_TNT", default = False, action = 'store_true', help = "TTbar")
options = parser.parse_args()

if(options.output[-1] != '/'): options.output +='/'

os.system("mkdir %s" %options.output)





#################################################################





compute_mjj_window(options)
print("Keep low %.0f keep high %.0f \n" % ( options.keep_mlow, options.keep_mhigh))
print("mjj low %.0f mjj high %.0f \n" % ( options.mjj_low, options.mjj_high))


options.keys = ['mjj', 'j1_features', 'j2_features', 'jet_kinematics', 'j1_AE_scores', 'j2_AE_scores']

do_both_js = False
options.randsort = True

j_label = "j1_"
opp_j_label = "j2_"


import time
t1 = time.time()
data, _ = load_dataset_from_options(options)


labeler_scores = data['j2_AE_scores']


print("Sig-rich region defined > %i percentile" %options.sig_cut)
print("Bkg-rich region defined < %i percentile" %options.bkg_cut)

#sig_region_cut = np.percentile(labeler_scores, options.sig_cut)
#bkg_region_cut = np.percentile(labeler_scores, options.bkg_cut)

#hardcode to avoid affect of signal
sig_region_cut = 3.553e-05
bkg_region_cut = 5.735e-06


if(options.do_TNT):
    print("Doing Tag N' Train regions")
    print("cut high %.3e, cut low %.3e " % (sig_region_cut, bkg_region_cut))
    data.make_Y_TNT(sig_region_cut = sig_region_cut, bkg_region_cut = bkg_region_cut, cut_var = labeler_scores, mjj_low = options.mjj_low, mjj_high = options.mjj_high,
            bkg_cut_type = options.TNT_bkg_cut)

    Y_label = "Y_TNT"
else:
    data.make_Y_mjj(options.mjj_low, options.mjj_high)
    Y_label = "Y_mjj"


t2 = time.time()
print("load time  %s " % (t2 -t1))


Y = data['label']

print_signal_fractions(Y, data[Y_label])

if(not options.nsubj_ratios):
    feature_names = ["jet mass", r"$\tau_2$", r"$\tau_3$", r"$\tau_4$", "LSF3", "DeepB", "nPFCands", "pt"]
    flabels = ["jetmass", "tau2", "tau3", "tau4", "LSF3", "DeepB", "nPFCands", "pt"]

else:
    feature_names = ["jet mass", r"$\tau_{21}$", r"$\tau_{32}$", r"$\tau_{43}$", "LSF3", "DeepB", "nPFCands", "pt"]
    flabels = ["jetmass", "tau21", "tau32", "tau43", "LSF3", "DeepB", "nPFCands", "pt"]


if(not options.keep_LSF):
    feature_names.remove("LSF3")

j1_pts = data['jet_kinematics'][:,2]
j2_pts = data['jet_kinematics'][:,6]
if(data.swapped_js):
    swaps = data['swapped_js'][()]
    j1_pts_temp = np.copy(j1_pts)
    j1_pts[swaps] = j2_pts[swaps]
    j2_pts[swaps] = j1_pts_temp[swaps]
    del j1_pts_temp


n_bins = 20
colors = ['b', 'green', 'red']
labels = ["All", "Sig-Rich Region", "Bkg-Rich Region",]


#signal only
is_sig =  (Y > 0.9).reshape(-1)
sig_region = (data[Y_label] == 1) & ( Y > 0.9).reshape(-1)
bkg_region = (data[Y_label] == 0 ) & ( Y > 0.9).reshape(-1)

ratio_range = 0.4

for i in range(len(feature_names)):
    feat_name = feature_names[i]

    if(feat_name == 'pt'): j1_feats = j1_pts
    else: j1_feats = data["j1_features"][:,i]
    j1_sig_region_feats = j1_feats[sig_region]
    j1_bkg_region_feats = j1_feats[bkg_region]
    j1_all_feats = j1_feats[is_sig]


    j1_bins, j1_ratio = make_ratio_histogram([j1_all_feats, j1_sig_region_feats, j1_bkg_region_feats], labels, 
            colors, 'J1 ' +feat_name, options.label, n_bins, ratio_range = ratio_range,  errors = True, 
                         normalize=True, save = True, fname=options.output + "j1_" + flabels[i] + ".png")


    if(feat_name == 'pt'): j2_feats = j2_pts
    else: j2_feats = data["j2_features"][:,i]
    j2_sig_region_feats = j2_feats[sig_region]
    j2_bkg_region_feats = j2_feats[bkg_region]
    j2_all_feats = j2_feats[is_sig]


    j2_bins, j2_ratio = make_ratio_histogram([j2_all_feats, j2_sig_region_feats, j2_bkg_region_feats], labels, 
            colors, 'j2 ' +feat_name, options.label, n_bins, ratio_range = ratio_range,  errors = True, 
                         normalize=True, save = True, fname=options.output + "j2_" + flabels[i] + ".png")

