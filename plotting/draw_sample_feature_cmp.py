import sys
sys.path.append('..')
from utils.TrainingUtils import *
from optparse import OptionParser
from optparse import OptionGroup




parser = input_options()
parser.add_argument("--draw_sig", default = False, action = 'store_true', help = "Include signal on feature comparison plots")
parser.add_argument("--do_ttbar", default = False, action = 'store_true', help = "TTbar")
options = parser.parse_args()

if(options.output[-1] != '/'): options.output +='/'

os.system("mkdir %s" %options.output)





#################################################################





compute_mjj_window(options)
print("Keep low %.0f keep high %.0f \n" % ( options.keep_mlow, options.keep_mhigh))
print("mjj low %.0f mjj high %.0f \n" % ( options.mjj_low, options.mjj_high))


options.keys = ['mjj', 'j1_features', 'j2_features', 'jet_kinematics']

do_TNT = False
if(options.do_ttbar):
    do_TNT = True
    do_both_js = False
    options.keep_mlow = 1400.
    options.keep_mhigh = 9999.
    options.ptsort = True

    j_label = "j1_"
    opp_j_label = "j2_"

elif(options.labeler_name != ''): #doing TNT
    do_TNT = True
    do_both_js = False
    options.randsort = True

    if(options.training_j == 1):
        j_label = "j1_"
        opp_j_label = "j2_"

    elif (options.training_j ==2):
        j_label = "j2_"
        opp_j_label = "j1_"


    if('auto' in options.labeler_name or 'AE' in options.labeler_name):
        options.keys.append(opp_j_label + "images")
        options.keys.append(j_label + "images")
        l_key = opp_j_label + 'images'
        if(options.randsort):
            l_key2 = j_label + "images"
            x_key2 = opp_j_label +  'features'
    else:
        if(options.use_images): l_key = opp_j_label +  'images'
        else: l_key = opp_j_label +  'features'
        if(options.randsort): 
            l_key2 = j_label +  'features'
            x_key2 = opp_j_label +  'features'


else:
    do_both_js = True


import time
t1 = time.time()
data, _ = load_dataset_from_options(options)

#print(data['mjj'][:5])
#print(data['j1_images'].shape)
#print(np.max(data['j1_images'][:5], axis = (1,2)))
#print(np.max(data['j2_images'][:5], axis = (1,2)))
#print(data['j1_features'][:5,0])
#print(data['j2_features'][:5,0])
#exit(1)


if(options.do_ttbar):
    print("Doing ttbar regions")
    filter_frac = data.make_Y_ttbar(data['j2_features'])
    Y_label = 'Y_ttbar'
    Y = data['label'] == -2


elif(do_TNT):
    print("Doing Tag N' Train regions")
    print("\n Loading labeling model from %s \n" % options.labeler_name)
    print("l_key",  l_key)
    Y_label = 'Y_TNT'
    labeler = tf.keras.models.load_model(options.labeler_name)

    labeler_scores = data.labeler_scores(labeler,  l_key)
    #labeler_scores = data['j2_features'][:,0]


    print("Sig-rich region defined > %i percentile" %options.sig_cut)
    print("Bkg-rich region defined < %i percentile" %options.bkg_cut)

    sig_region_cut = np.percentile(labeler_scores, options.sig_cut)
    bkg_region_cut = np.percentile(labeler_scores, options.bkg_cut)

    print("cut high %.3e, cut low %.3e " % (sig_region_cut, bkg_region_cut))

    data.make_Y_TNT(sig_region_cut = sig_region_cut, bkg_region_cut = bkg_region_cut, cut_var = labeler_scores, mjj_low = options.mjj_low, mjj_high = options.mjj_high,
            bkg_cut_type = options.TNT_bkg_cut)


else:
    print("Doing CWoLa regions")
    Y_label = 'Y_mjj'
    data.make_Y_mjj(options.mjj_low, options.mjj_high)

t2 = time.time()
print("load time  %s " % (t2 -t1))


Y = data['label']

print_signal_fractions(Y, data[Y_label])
minor_bkg = (Y < 0).reshape(-1)
print("Minor bkg frac %.3f in SR %.3f in SB" % (np.mean(minor_bkg & (data[Y_label] > 0.9)), np.mean(minor_bkg & (data[Y_label] < 0.1))))


#print(data['jet_kinematics'][:10])
#print(data['j1_features'][:10,0], data['j2_features'][:10,0])




data.make_ptrw(Y_label, use_weights = not options.no_sample_weights, save_plots = True, plot_dir = options.output)

if(not options.nsubj_ratios):
    feature_names = ["jet mass", r'$\tau_1$', r"$\tau_2$", r"$\tau_3$", r"$\tau_4$", "LSF3", "DeepB", "nPFCands", "pt"]
    flabels = ["jetmass","tau1", "tau2", "tau3", "tau4", "LSF3", "DeepB", "nPFCands", "pt"]

else:
    feature_names = ["jet mass", r'$\tau_1$', r"$\tau_{21}$", r"$\tau_{32}$", r"$\tau_{43}$", "LSF3", "DeepB", "nPFCands", "pt"]
    flabels = ["jetmass","tau1", "tau21", "tau32", "tau43", "LSF3", "DeepB", "nPFCands", "pt"]

if(not options.keep_LSF):
    feature_names.remove("LSF3")

n_bins = 20
colors = ['b', 'green']
labels = ["Signal Region", "Bkg Region"]


j1_pts = data['jet_kinematics'][:,2]
j2_pts = data['jet_kinematics'][:,6]
if(data.swapped_js):
    swaps = data['swapped_js'][()]
    j1_pts_temp = np.copy(j1_pts)
    j1_pts[swaps] = j2_pts[swaps]
    j2_pts[swaps] = j1_pts_temp[swaps]
    del j1_pts_temp


#j2_pt_cut = (j2_pts > 400.) & (j2_pts < 425.)
#j2_pt_cut = (j2_pts > -10.)

#sig_region = (data['j1_features'][:,0] > 100.) & j2_pt_cut
#bkg_region = (data['j1_features'][:,0] < 100.) & j2_pt_cut


sig_region = data[Y_label] == 1
bkg_region = data[Y_label] == 0

bkg_mjjs = data['mjj'][bkg_region]


print("\n There are %i events in the signal region and %i in the bkg region \n" % (np.sum(sig_region), np.sum(bkg_region)))
if(not options.do_ttbar):
    print("\n There are %i events in the low-mjj bkg region and %i in the high-mjj bkg region \n" % (np.sum(bkg_mjjs < options.mjj_low), np.sum(bkg_mjjs > options.mjj_high )))




true_bkg = (Y < 0.1).reshape(-1)
true_sig = (Y > 0.9).reshape(-1)
if(not options.draw_sig):
    true_bkg = true_sig = (Y > -99999.).reshape(-1)


#j1_sig_ms = data['j1_features'][:,0] [true_sig]
#j2_sig_ms = data['j2_features'][:,0] [true_sig]
#print(j1_sig_ms[:10])
#print(j2_sig_ms[:10])
#print(np.mean(j1_sig_ms > j2_sig_ms))

if(options.draw_sig):
    print("Drawing based on %.0f signal events" % np.sum(sig_region & true_sig))

j1_ptrw_sig_region_weights = data['j1_ptrw'][sig_region & true_bkg]
j1_ptrw_bkg_region_weights = data['j1_ptrw'][bkg_region & true_bkg] 
j1_ptrw_signal = data['j1_ptrw'][sig_region & true_sig]

j2_ptrw_sig_region_weights = data['j2_ptrw'][sig_region & true_bkg] 
j2_ptrw_bkg_region_weights = data['j2_ptrw'][bkg_region & true_bkg]
j2_ptrw_signal = data['j2_ptrw'][sig_region & true_sig]


if(not options.no_sample_weights):
    sig_region_weights = data['weight'][sig_region & true_bkg]
    bkg_region_weights = data['weight'][bkg_region & true_bkg]

    weights_noptrw = [sig_region_weights, bkg_region_weights]
else:
    weights_noptrw = None

ratio_range = 0.4


sig_region_mjj = data['mjj'][sig_region & true_bkg]
bkg_region_mjj = data['mjj'][bkg_region & true_bkg]


j1_bins, j1_ratio = make_ratio_histogram([sig_region_mjj, bkg_region_mjj], labels, 
        colors, 'Mjj', "", n_bins, ratio_range = ratio_range, weights = weights_noptrw, errors = True, 
        normalize=True, save = True, fname=options.output + 'mjj' + ".png")


mass_greater = np.mean(data['j1_features'][:,0] > data['j2_features'][:,0])
print("J1 mass greater than J2 mass frac : %.3f \n" % mass_greater)


for i in range(len(feature_names)):
    feat_name = feature_names[i]

    if(do_both_js or options.training_j == 1 or True):
        if(feat_name == 'pt'): j1_feats = j1_pts
        else: j1_feats = data["j1_features"][:,i]
        #if('LSF' in feat_name):
            #print(j1_feats[:50])

        j1_sig_region_feats = j1_feats[sig_region & true_bkg]
        j1_bkg_region_feats = j1_feats[bkg_region & true_bkg]
        j1_signal_feats = j1_feats[sig_region & true_sig]

        if(options.draw_sig):
            extras = [(j1_signal_feats, j1_ptrw_signal, "Signal")]
        else:
            extras = None

        j1_bins, j1_ratio = make_ratio_histogram([j1_sig_region_feats, j1_bkg_region_feats], labels, 
                colors, 'J1 ' +feat_name, "No pt Reweighting", n_bins, ratio_range = ratio_range, weights = weights_noptrw, errors = True, 
                            extras = extras, normalize=True, save = True, fname=options.output + "j1_" + flabels[i] + ".png")

        j1_bins, j1_ratio = make_ratio_histogram([j1_sig_region_feats, j1_bkg_region_feats], labels, 
                colors, 'J1 ' +feat_name, "With pt Reweighting", n_bins, ratio_range = ratio_range, weights = [j1_ptrw_sig_region_weights, j1_ptrw_bkg_region_weights], errors = True,
                            extras = extras, normalize=True, save = True, fname=options.output + "j1_" + flabels[i] + "_ptrw" + ".png")

    if(do_both_js or options.training_j == 2 or True):
        if(feat_name == 'pt'): j2_feats = j2_pts
        else: j2_feats = data["j2_features"][:,i]

        j2_sig_region_feats = j2_feats[sig_region & true_bkg]
        j2_bkg_region_feats = j2_feats[bkg_region & true_bkg]
        j2_signal_feats = j2_feats[sig_region & true_sig]

        if(options.draw_sig):
            extras = [(j2_signal_feats, j2_ptrw_signal, "Signal")]
        else:
            extras = None
        j2_bins, j2_ratio = make_ratio_histogram([j2_sig_region_feats, j2_bkg_region_feats], labels, 
                colors, 'J2 ' +feat_name, "No pt Reweighting", n_bins, ratio_range = ratio_range, weights = weights_noptrw, errors = True, 
                            extras = extras, normalize=True, save = True, fname=options.output + "j2_" + flabels[i] + ".png")


        j2_bins, j2_ratio = make_ratio_histogram([j2_sig_region_feats, j2_bkg_region_feats], labels, 
                colors, 'J2 ' +feat_name, "With pt Reweighting", n_bins, ratio_range = ratio_range, weights = [j2_ptrw_sig_region_weights, j2_ptrw_bkg_region_weights], errors = True,
                            extras = extras, normalize=True, save = True, fname=options.output + "j2_" + flabels[i] + "_ptrw" + ".png")

del data

