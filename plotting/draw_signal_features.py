import sys
sys.path.append('..')
from utils.TrainingUtils import *
from optparse import OptionParser
from optparse import OptionGroup




parser = input_options()
options = parser.parse_args()

if(options.output[-1] != '/'): options.output +='/'

os.system("mkdir %s" %options.output)





#################################################################







options.keys = ['mjj', 'j1_features', 'j2_features', 'jet_kinematics']

do_both_js = True


import time
t1 = time.time()
data, _ = load_dataset_from_options(options)

Y = data['label']

if(not options.nsubj_ratios):
    feature_names = ["jet mass", r"$\tau_2$", r"$\tau_3$", r"$\tau_4$", "LSF3", "DeepB", "nPFCands"]
    flabels = ["jetmass", "tau2", "tau3", "tau4", "LSF3", "DeepB", "nPFCands", "pt"]

else:
    feature_names = ["jet mass", r"$\tau_{21}$", r"$\tau_{32}$", r"$\tau_{43}$", "LSF3", "DeepB", "nPFCands"]
    flabels = ["jetmass", "tau21", "tau32", "tau43", "LSF3", "DeepB", "nPFCands", "pt"]

if(not options.keep_LSF):
    feature_names.remove("LSF3")



n_bins = 20
colors = ['b', 'red']
labels = ["background", "signal"]
normalize = True




#j2_pt_cut = (j2_pts > 400.) & (j2_pts < 425.)
#j2_pt_cut = (j2_pts > -10.)

#sig_region = (data['j1_features'][:,0] > 100.) & j2_pt_cut
#bkg_region = (data['j1_features'][:,0] < 100.) & j2_pt_cut



true_bkg = (Y < 0.1).reshape(-1)
true_sig = (Y > 0.9).reshape(-1)

print("Drawing based on %.0f signal events" % np.sum(true_sig))




bkg_mjj = data['mjj'][true_bkg]
signal_mjj = data['mjj'][true_sig]


make_histogram([bkg_mjj, signal_mjj], labels, 
        colors, 'Mjj', "", n_bins,  logy = True,
        normalize=normalize, fname=options.output + 'mjj' + ".png")


for i in range(len(feature_names)):
    feat_name = feature_names[i]

    if(do_both_js or options.training_j == 1 or True):
        j1_feats = data["j1_features"][:,i]
        #if('LSF' in feat_name):
            #print(j1_feats[:50])

        j1_bkg_feats = j1_feats[true_bkg]
        j1_sig_feats = j1_feats[true_sig]


        make_histogram([j1_bkg_feats, j1_sig_feats], labels, colors, 'J1 ' +feat_name,  n_bins,  
                    normalize=normalize, fname=options.output + "j1_" + flabels[i] + ".png")


    if(do_both_js or options.training_j == 2 or True):
        j2_feats = data["j2_features"][:,i]

        j2_bkg_feats = j2_feats[true_bkg]
        j2_sig_feats = j2_feats[true_sig]

        make_histogram([j2_bkg_feats, j2_sig_feats], labels, colors, 'j2 ' +feat_name,  n_bins,  
                    normalize=normalize, fname=options.output + "j2_" + flabels[i] + ".png")




del data

