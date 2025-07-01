import sys
sys.path.append('..')
from utils.TrainingUtils import *
from optparse import OptionParser
from optparse import OptionGroup

def check_new_lund(options, f_old, f_new, frac_const = 5):
    #frac_const = 10

    if(options.output[-1] != '/'): options.output +='/'

    os.system("mkdir %s" %options.output)


    compute_mjj_window(options)
    print("Keep low %.0f keep high %.0f \n" % ( options.keep_mlow, options.keep_mhigh))
    print("mjj low %.0f mjj high %.0f \n" % ( options.mjj_low, options.mjj_high))

    options.keys = ['mjj', 'j1_features', 'j2_features', 'jet_kinematics']

    options.batch_start = 0
    options.batch_stop = 39
    options.randsort = False

    if(options.sig_per_batch > 0):
        num_sig_tot = options.sig_per_batch * 40
        #scale up amount of sig so have smaller uncertainties
        num_sig_tot *= frac_const**2
    else:
        num_sig_tot = -1



    norm = True

    import time
    t1 = time.time()
    options.sig_file = f_old
    sig_only_data_old = load_signal_file(options)

    options.sig_file = f_new
    sig_only_data_new = load_signal_file(options)

    t2 = time.time()
    print("load time  %s " % (t2 -t1))

    j_labels = ["j1", "j2"] if not options.randsort else ["j1"]


    mjj_diffs = np.mean(sig_only_data_old["jet_kinematics"][:,0] - sig_only_data_new["jet_kinematics"][:,0])
    tau21_diffs = np.mean(sig_only_data_old["j1_features"][:,1] - sig_only_data_new["j1_features"][:,1])
    LP_diffs = np.mean(sig_only_data_old["lund_weights"][:] - sig_only_data_new["lund_weights"][:])
    print("Diffs: Mjj %.3f, tau21 %.3f, LP weights %.3f" % (mjj_diffs, tau21_diffs, LP_diffs))





    feature_names = ["jet mass", r"$\tau_{21}$", r"$\tau_{32}$", r"$\tau_{43}$", "LSF3", "DeepB", "nPFCands", "mjj"]
    flabels = ["jetmass", "tau21", "tau32", "tau43", "LSF3", "DeepB", "nPFCands", "mjj"]

    #clip outliers
    cutoff = 1.

    n_bins = 10
    #colors = ['black', 'blue', 'red']
    #labels = ["Nominal", "Sys. Up", "Sys. Down"]
    colors = ['blue', 'black', 'red',]
    labels = ["Orig Lund Weights", "Nominal", "V2 Lund Weights"]

    if(num_sig_tot >0):
        labels.append("Signal Injection. Stat. Unc. / %.0f" % frac_const)

    ratio_range = 0.1
    if(num_sig_tot > 0):
        ratio_range = 0.15

    relevant_uncs = set()

    mjj  = sig_only_data_old['jet_kinematics'][:,0]
    mjj_max = np.percentile(mjj, 100. - cutoff)
    mjj_min = np.percentile(mjj, cutoff)
    mjj = np.clip(mjj, mjj_min, mjj_max)

    for jl in j_labels:
        j_m_nom =   sig_only_data_old["%s_features" % jl][:,0]
        m_max = np.percentile(j_m_nom, 100. - cutoff)
        m_min = np.percentile(j_m_nom, cutoff)
        j_m_nom = np.clip(j_m_nom, m_min, m_max)

        sig_weights_nom = sig_only_data_old['sys_weights'][:,0]
        sig_weights_old = sig_only_data_old['sys_weights'][:,0]
        sig_weights_old *= sig_only_data_old['lund_weights'][:]

        sig_weights_new = sig_only_data_new['sys_weights'][:,0]
        sig_weights_new *= sig_only_data_new['lund_weights'][:]

        lund_sys_up = sig_only_data_new['lund_weights_sys_var'][:,0]
        lund_sys_down = sig_only_data_new['lund_weights_sys_var'][:,1]

        lund_bquark_up = sig_only_data_new['lund_weights_sys_var'][:,2]
        lund_bquark_down = sig_only_data_new['lund_weights_sys_var'][:,3]

        lund_prongs_up = sig_only_data_new['lund_weights_sys_var'][:,4]
        lund_prongs_down = sig_only_data_new['lund_weights_sys_var'][:,5]

        lund_unclust_up = sig_only_data_new['lund_weights_sys_var'][:,6]
        lund_unclust_down = sig_only_data_new['lund_weights_sys_var'][:,7]

        lund_distort_up = sig_only_data_new['lund_weights_sys_var'][:,8]
        lund_distort_down = sig_only_data_new['lund_weights_sys_var'][:,9]

        print("nom", np.mean(sig_weights_old))

        weights = [sig_weights_old, sig_weights_nom, sig_weights_new]

        for i in range(len(feature_names)):
            feat_name = feature_names[i]

            if(feat_name == 'mjj'): feat = mjj
            else: 
                feat = sig_only_data_old["%s_features" %jl][:,i]
                feat = np.clip(feat, np.percentile(feat, cutoff), np.percentile(feat, 100. - cutoff))



            ns, bins, ratios, frac_unc = make_multi_ratio_histogram([feat, feat, feat], labels, 
                    colors, feat_name, feat_name, n_bins, ratio_range = ratio_range, weights = weights, errors = False, 
                normalize=norm, save = True, fname=options.output + jl + '_' + flabels[i]  + ".png", unc_band_norm = num_sig_tot)

            #diff = abs(ratios[0] - 1.0)
            #JS = (JSD(ns[0], ns[1])  + JSD(ns[0], ns[2])) / 2. 

            #if(any(diff > frac_unc)):
            #    print(sys, JS, ratios[0])
            #    relevant_uncs.add(sys)

    print("Relevant systematics were :", relevant_uncs)
    return relevant_uncs

if(__name__ == "__main__"):
    parser = input_options()
    options = parser.parse_args()
    check_new_lund(options, options.sig_file, options.sig2_file)
