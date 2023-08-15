import sys
sys.path.append('..')
from utils.TrainingUtils import *
from optparse import OptionParser
from optparse import OptionGroup









def draw_sys_variations(options, frac_const = 5):
    #frac_const = 10

    if(options.output[-1] != '/'): options.output +='/'

    os.system("mkdir %s" %options.output)


    compute_mjj_window(options)
    print("Keep low %.0f keep high %.0f \n" % ( options.keep_mlow, options.keep_mhigh))
    print("mjj low %.0f mjj high %.0f \n" % ( options.mjj_low, options.mjj_high))

    options.keys = ['mjj', 'j1_features', 'j2_features', 'jet_kinematics']

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

    j_labels = ["j1", "j2"] if not options.randsort else ["j1"]





    feature_names = ["jet mass", r"$\tau_{21}$", r"$\tau_{32}$", r"$\tau_{43}$", "LSF3", "DeepB", "nPFCands", "mjj"]
    flabels = ["jetmass", "tau21", "tau32", "tau43", "LSF3", "DeepB", "nPFCands", "mjj"]

    #clip outliers
    cutoff = 1.

    n_bins = 10
    colors = ['black', 'blue', 'red']
    labels = ["Nominal", "Sys. Up", "Sys. Down"]

    if(num_sig_tot >0):
        labels.append("Signal Injection. Stat. Unc. / %.0f" % frac_const)

    ratio_range = 0.1
    if(num_sig_tot > 0):
        ratio_range = 0.15

    relevant_uncs = set()

    mjj  = sig_only_data['jet_kinematics'][:,0]
    mjj_max = np.percentile(mjj, 100. - cutoff)
    mjj_min = np.percentile(mjj, cutoff)
    mjj = np.clip(mjj, mjj_min, mjj_max)


    for jl in j_labels:
        j_m_nom =   sig_only_data["%s_features" % jl][:,0]
        m_max = np.percentile(j_m_nom, 100. - cutoff)
        m_min = np.percentile(j_m_nom, cutoff)
        j_m_nom = np.clip(j_m_nom, m_min, m_max)

        sig_weights_nom = sig_only_data['sys_weights'][:,0]
        if(options.lund_weights): sig_weights_nom *= sig_only_data['lund_weights'][:]

        print("nom", np.mean(sig_weights_nom))



        for jme_var in JME_vars_clean:
            up_str = jme_var + "_up"
            down_str = jme_var + "_down"


            j_m_up =   np.clip(sig_only_data["%s_JME_vars" %jl][:, JME_vars_map["m_" + up_str]], m_min, m_max)
            j_m_down = np.clip(sig_only_data["%s_JME_vars" %jl][:, JME_vars_map["m_" + down_str]], m_min, m_max)



            ns, bins, ratios, frac_unc = make_multi_ratio_histogram([j_m_nom, j_m_up, j_m_down], labels, 
                    colors, "Jet Mass", jme_var + " : Jet Mass", n_bins, ratio_range = ratio_range, weights = [sig_weights_nom, sig_weights_nom, sig_weights_nom], errors = False, 
                normalize=norm, save = True, fname=options.output + jme_var + '_mass'  + ".png", unc_band_norm = num_sig_tot)


            JS = (JSD(ns[0], ns[1])  + JSD(ns[0], ns[2])) / 2. 

            diff = abs(ratios[0] - 1.0)



            if(any(diff > frac_unc )):
                print(jme_var, JS, ratios[0])
                relevant_uncs.add(jme_var)

            if('JE' in jme_var): #also correct pt (JES and JER)

                j1_pt_up =   sig_only_data["j1_JME_vars"][:, JME_vars_map["pt_" + up_str]]
                j1_pt_down = sig_only_data["j1_JME_vars"][:, JME_vars_map["pt_" + down_str]]


                j2_pt_up =   sig_only_data["j2_JME_vars"][:, JME_vars_map["pt_" + up_str]]
                j2_pt_down = sig_only_data["j2_JME_vars"][:, JME_vars_map["pt_" + down_str]]

                sig_jet_kinematics = sig_only_data['jet_kinematics'][:]
                sig_jet_kinematics [:, 2] = j1_pt_up
                sig_jet_kinematics [:, 6] = j2_pt_up
                mjj_up = np.clip(mjj_from_4vecs(sig_jet_kinematics[:, 2:6], sig_jet_kinematics[:, 6:10]), mjj_min, mjj_max)


                sig_jet_kinematics [:, 2] = j1_pt_down
                sig_jet_kinematics [:, 6] = j2_pt_down
                mjj_down = np.clip(mjj_from_4vecs(sig_jet_kinematics[:, 2:6], sig_jet_kinematics[:, 6:10]), mjj_min, mjj_max)

                ns, bins, ratios, frac_unc = make_multi_ratio_histogram([mjj, mjj_up, mjj_down], labels, 
                    colors, "Mjj", jme_var + " : Mjj", n_bins, ratio_range = ratio_range, weights = [sig_weights_nom, sig_weights_nom, sig_weights_nom], errors = False, 
                normalize=norm, save = True, fname=options.output + jme_var + '_mjj'  + ".png", unc_band_norm = num_sig_tot)

                diff = abs(ratios[0] - 1.0)
                JS = (JSD(ns[0], ns[1])  + JSD(ns[0], ns[2])) / 2. 

                if(any(diff > frac_unc)):
                    print(jme_var, JS, ratios[0])
                    relevant_uncs.add(jme_var)




        #sys_list_clean.add('lund_nom')
        for sys in sys_list_clean:
            if("JE" in sys or "JM" in sys): continue

            sys_up = sys + "_up"
            sys_down = sys + "_down"

            #if(sys == 'lund_nom'):
            #    up_weights = sig_weights_nom * sig_only_data['lund_weights'][:]
            #    down_weights = sig_weights_nom * sig_only_data['lund_weights'][:]

            if("lund" in sys):
                if(options.lund_weights):
                    up_weight_idx = lund_vars_map[sys_up]
                    up_weights = sig_weights_nom * sig_only_data['lund_weights_sys_var'][:,up_weight_idx]
                    down_weight_idx = lund_vars_map[sys_down]
                    down_weights = sig_weights_nom * sig_only_data['lund_weights_sys_var'][:,down_weight_idx]
                else: 
                    continue

            else:
                up_weight_idx = sys_weights_map[sys_up]
                up_weights = sig_weights_nom *  sig_only_data['sys_weights'][:,up_weight_idx]
                down_weight_idx = sys_weights_map[sys_down]
                down_weights = sig_weights_nom *  sig_only_data['sys_weights'][:,down_weight_idx]

            print(sys, np.mean(up_weights), np.mean(down_weights))


            for i in range(len(feature_names)):
                feat_name = feature_names[i]

                if(feat_name == 'mjj'): feat = mjj
                else: 
                    feat = sig_only_data["%s_features" %jl][:,i]
                    feat = np.clip(feat, np.percentile(feat, cutoff), np.percentile(feat, 100. - cutoff))



                ns, bins, ratios, frac_unc = make_multi_ratio_histogram([feat, feat, feat], labels, 
                        colors, feat_name, sys + " : " + feat_name, n_bins, ratio_range = ratio_range, weights = [sig_weights_nom, up_weights, down_weights], errors = False, 
                    normalize=norm, save = True, fname=options.output + jl + "_" + sys + '_' + flabels[i]  + ".png", unc_band_norm = num_sig_tot)

                diff = abs(ratios[0] - 1.0)
                JS = (JSD(ns[0], ns[1])  + JSD(ns[0], ns[2])) / 2. 

                if(any(diff > frac_unc)):
                    print(sys, JS, ratios[0])
                    relevant_uncs.add(sys)

    print("Relevant systematics were :", relevant_uncs)
    return relevant_uncs

if(__name__ == "__main__"):
    parser = input_options()
    options = parser.parse_args()
    draw_sys_variations(options)
