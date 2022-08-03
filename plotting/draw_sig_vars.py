import sys
sys.path.append('..')
from utils.TrainingUtils import *
from optparse import OptionParser
from optparse import OptionGroup









def draw_sig_vars(options):
    #frac_const = 10

    if(options.output[-1] != '/'): options.output +='/'

    os.system("mkdir %s" %options.output)


    compute_mjj_window(options)
    print("Keep low %.0f keep high %.0f \n" % ( options.keep_mlow, options.keep_mhigh))
    print("mjj low %.0f mjj high %.0f \n" % ( options.mjj_low, options.mjj_high))

    options.keys = ['mjj', 'j1_features', 'j2_features', 'jet_kinematics', 'event_info']

    options.batch_start = 0
    options.batch_stop = 39

    if(options.sig_per_batch > 0):
        num_sig_tot = options.sig_per_batch * 40
        #scale up amount of sig so have smaller uncertainties
        num_sig_tot *= frac_const**2
    else:
        num_sig_tot = -1



    norm = True

    import time
    t1 = time.time()
    sig_only_data = load_signal_file(options)

    t2 = time.time()
    print("load time  %s " % (t2 -t1))





    feature_names = ["jet mass", r'$\tau_1$', r"$\tau_{21}$", r"$\tau_{32}$", r"$\tau_{43}$", "LSF3", "DeepB", "nPFCands"]
    flabels = ["jetmass","tau1", "tau21", "tau32", "tau43", "LSF3", "DeepB", "nPFCands"]

    #clip outliers
    cutoff = 1.

    n_bins = 15
    colors = ['black']
    labels = ["Nominal"]

    mjj  = sig_only_data['jet_kinematics'][:,0]



    make_histogram([mjj], labels, colors,  "Mjj", "Mjj", n_bins, fname = options.output + "mjj.png")

    jet3_pt = np.clip(sig_only_data['jet_kinematics'][:,10], 0., 1000.)
    make_histogram([jet3_pt], labels, colors,  "Jet3 pt", "jet3 pt", n_bins, fname = options.output + "j3_pt.png")
    #n_jets = sig_only_data['event_info'][:,7]
    #make_histogram([n_jets], labels, colors,  "nJets", "nJets", n_bins, fname = options.output + "njets.png")


    for i in range(len(feature_names)):
        feat_name = feature_names[i]

        feat1 = sig_only_data["j1_features"][:,i]
        feat2 = sig_only_data["j2_features"][:,i]

        make_histogram([feat1], labels, colors, feat_name, "J1", n_bins, fname = options.output + "j1_" + flabels[i] + ".png")
        make_histogram([feat2], labels, colors, feat_name, "J2", n_bins, fname = options.output + "j2_" + flabels[i] + ".png")

    return None

if(__name__ == "__main__"):
    parser = input_options()
    options = parser.parse_args()
    draw_sig_vars(options)
