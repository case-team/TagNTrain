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

options.batch_start = 0
options.batch_stop = 39

data = load_signal_file(options)

feature_names = ["jet mass", r"$\tau_{21}$", r"$\tau_{32}$", r"$\tau_{43}$", "LSF3", "DeepB", "nPFCands"]
flabels = ["jetmass", "tau21", "tau32", "tau43", "LSF3", "DeepB", "nPFCands"]


nbins = 20
mbins = np.linspace(options.mjj_sig * 0.7, options.mjj_sig *1.3, nbins)

j1_avgs = [[] for fname in feature_names]
j2_avgs = [[] for fname in feature_names]
mjj_centers = []
mjj = data['jet_kinematics'][:,0]
make_histogram([mjj], ['mjj'], ['blue'], "Mjj",  num_bins = nbins,  logy = True, h_range = (options.mjj_sig * 0.7, options.mjj_sig * 1.3), normalize=True, fname=options.output + "mjj.png" )

for i in range(len(mbins)-1):
    mlow = mbins[i]
    mhigh = mbins[i+1]
    mjj_center = (mhigh + mlow) / 2.0
    mjj_centers.append(mjj_center)
    mask = (mjj > mlow ) & (mjj < mhigh)
    for j,feat_name in enumerate(feature_names):
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

