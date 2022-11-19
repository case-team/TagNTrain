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






options.mjj_low = -1
options.mjj_high = -1
options.keys = ['mjj', 'j1_features', 'j2_features', 'jet_kinematics']

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


t2 = time.time()
print("load time  %s " % (t2 -t1))


Y = data['label']
mjj = data['jet_kinematics'][:,0]
j1_pts = data['jet_kinematics'][:,2]
j2_pts = data['jet_kinematics'][:,6]
minor_bkg = (Y < 0).reshape(-1)



if(not options.nsubj_ratios):
    feature_names = ["jet mass", r'$\tau_1$', r"$\tau_2$", r"$\tau_3$", r"$\tau_4$", "LSF3", "DeepB", "nPFCands", "pt"]
    flabels = ["jetmass","tau1", "tau2", "tau3", "tau4", "LSF3", "DeepB", "nPFCands", "pt"]

else:
    feature_names = ["jet mass", r'$\tau_1$', r"$\tau_{21}$", r"$\tau_{32}$", r"$\tau_{43}$", "LSF3", "DeepB", "nPFCands", "pt"]
    flabels = ["jetmass","tau1", "tau21", "tau32", "tau43", "LSF3", "DeepB", "nPFCands", "pt"]

if(not options.keep_LSF):
    feature_names.remove("LSF3")

mbins = mass_bins1

j1_avgs = [[] for fname in feature_names]
j2_avgs = [[] for fname in feature_names]
mjj_centers = []

for i in range(len(mbins)-1):
    mlow = mbins[i]
    mhigh = mbins[i+1]
    mjj_center = (mhigh + mlow) / 2.0
    mjj_centers.append(mjj_center)
    mask = (mjj > mlow ) & (mjj < mhigh)
    for j,feat_name in enumerate(feature_names):
        if(feat_name == 'pt'): 
            j1_feats = j1_pts
            j2_feats = j2_pts
        else: 
            j1_feats = data["j1_features"][:,j]
            j2_feats = data["j2_features"][:,j]

        j1_avg = np.mean(j1_feats[mask])
        j2_avg = np.mean(j2_feats[mask])

        j1_avgs[j].append(j1_avg)
        j2_avgs[j].append(j2_avg)

labels = ["J1", "J2"]
colors = ["red", "blue"]
for j,feat in enumerate(feature_names):
    make_graph([mjj_centers, mjj_centers], [j1_avgs[j], j2_avgs[j]], labels, colors, ["Mjj", feat], 
            fname = options.output + "mjj_variation_" + flabels[j] + ".png" )



