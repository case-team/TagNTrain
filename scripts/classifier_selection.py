import sys
sys.path.append('..')
from utils.TrainingUtils import *
import h5py


def get_features(data, systematic = ""):
    j1_images = j2_images = jj_images = j1_dense_inputs = j2_dense_inputs = jj_dense_inputs = None

    if('j1_images' in data.keys): 
        j1_images = data['j1_images']
        j2_images = data['j2_images']
    if('jj_images' in data.keys): jj_images = data['jj_images']
    if('j1_features' in data.keys):
        j1_dense_inputs = data['j1_features']
        j2_dense_inputs = data['j2_features']

        #modify features for JME corrections (for signals only)
        if(len(systematic) > 0 and systematic in JME_vars):
            m_idx = JME_vars_map["m_" + systematic]
            j1_m_corr = data["j1_JME_vars"][:, m_idx]
            j2_m_corr = data["j2_JME_vars"][:, m_idx]


            j1_dense_inputs[:, 0] = j1_m_corr
            j2_dense_inputs[:, 0] = j2_m_corr


    if('jj_dense_inputs' in data.keys):
        jj_dense_inputs = data['jj_features']

    return j1_images, j2_images, j1_dense_inputs, j2_dense_inputs, jj_images, jj_dense_inputs
    


def classifier_selection(options):
    print("\n")
    compute_mjj_window(options)
    options.keep_mlow = options.keep_mhigh = -1
    print(options.__dict__)

    #model types: 0 CNN (one jet), 1 auto encoder, 2 dense (one jet), 3 CNN (both jets), 4 dense (both jets), 5 is VAE 
    if(len(options.effs) == 0):
        print("Must input at least 1 efficiency value!")

    options.keys = ['mjj', 'event_info']
    if(options.model_type == 0 or options.model_type == 1):
        options.keys += ['j1_images', 'j2_images']
    if(options.model_type == 3 ):
        options.keys.append('jj_images' )
    if(options.model_type == 2):
        options.keys += ['j1_features', 'j2_features']
    if(options.model_type == 4):
        options.keys.append('jj_features' )
    #keys = ["j1_images", "j2_images", "jj_images", "j1_features", "j2_features", "jj_features", 'mjj']

    data, _ = load_dataset_from_options(options)
    sig_only_data = None
    if(len(options.sig_file) > 0):
        sig_only_data = load_signal_file(options)

    Y = data['label'].reshape(-1)
    mjj = data['mjj']

    event_num = data['event_info'][:,0]

    j1_images, j2_images, j1_dense_inputs, j2_dense_inputs, jj_images, jj_dense_inputs = get_features(data)

    if(sig_only_data is not None):
        j1_sig_images, j2_sig_images, j1_dense_sig_inputs, j2_dense_sig_inputs, jj_sig_images, jj_dense_sig_inputs = get_features(sig_only_data, systematic = options.sig_sys)

    batch_size = 1024
    sig_effs = []
    bkg_effs = []
    overall_effs = []

    if(len(options.effs) ==1 and options.effs[0] == 100.):
        jj_scores = None

    else:
        if('sig_idx' in options.labeler_name): 
            f = options.labeler_name.format(sig_idx = options.sig_idx)
        else:
            f = options.labeler_name

        print("Using model %s" % f)
        if(options.model_type <= 2 or options.model_type == 5): #classifier on each jet

            j1_score, j2_score = get_jet_scores("", f, options.model_type, j1_images, j2_images, j1_dense_inputs, j2_dense_inputs, num_models = options.num_models)
            j1_QT = QuantileTransformer(copy = True)
            j1_qs = j1_QT.fit_transform(j1_score.reshape(-1,1)).reshape(-1)
            j2_QT = QuantileTransformer(copy = True)
            j2_qs = j2_QT.fit_transform(j2_score.reshape(-1,1)).reshape(-1)

            #combine scores into one
            jj_scores = combine_scores(j1_qs, j2_qs, options.score_comb)

            if(sig_only_data is not None):
                j1_sig_score, j2_sig_score = get_jet_scores("", f, options.model_type, j1_sig_images, j2_sig_images, j1_dense_sig_inputs, j2_dense_sig_inputs, num_models = options.num_models)
                j1_sig_qs = j1_QT.transform(j1_sig_score.reshape(-1,1)).reshape(-1)
                j2_sig_qs = j2_QT.transform(j2_sig_score.reshape(-1,1)).reshape(-1)
                jj_sig_scores = combine_scores(j1_sig_qs, j2_sig_qs, options.score_comb)


        else:
            jj_scores = get_jj_scores("", f, options.model_type, jj_images, jj_dense_inputs, num_models = options.num_models)

        if(options.do_roc):
            in_window_and_nominor_bkg = (mjj > options.mjj_low) & (mjj < options.mjj_high) & (Y > -0.1) 
            QCD = (mjj > options.mjj_low) & (mjj < options.mjj_high) & (Y > -0.1) & (Y < 0.1)
            sig = (mjj > options.mjj_low) & (mjj < options.mjj_high) & (Y > 0.1)
            QCD_scores = jj_scores[QCD]
            if(sig_only_data is not None): sig_scores = jj_sig_scores
            else: sig_scores = jj_scores[sig]
            labels = np.append([0.] * QCD_scores.shape[0], [1.] * sig_scores.shape[0])
            scores = np.append(QCD_scores, sig_scores)
            bkg_eff_roc, sig_eff_roc, thresholds = roc_curve(labels, scores, drop_intermediate = True)
            print("auc", auc(bkg_eff_roc, sig_eff_roc))



    for eff in options.effs:

        #sideband, weighted_sideband, central or overall
        #eff_definition = 'central'
        eff_definition = 'weighted_sideband'
        use_sidebands = True
        weighted_sidebands = True

        if(options.mbin > 0):
            sb_mlow, mjj_low, mjj_high, sb_mhigh = lookup_mjj_bins(options.mbin)
        else:

            window_size = (options.mjj_high - options.mjj_low)/2.
            window_frac = window_size / ((options.mjj_high + options.mjj_low)/ 2.)

            window_low_size = window_frac*options.mjj_low / (1 + window_frac)
            window_high_size = window_frac*options.mjj_high / (1 - window_frac)
            sb_mlow = options.mjj_low - window_low_size
            sb_mhigh = options.mjj_high + window_high_size
            mjj_low = options.mjj_low
            mjj_high = options.mjj_high


        print("Will select events with efficiency %.3f" % eff)
        percentile_cut = 100. - eff
        if('{eff}' in options.output):
            output_name = options.output.format(eff = eff)
        else:
            output_name = options.output

        if(eff == 100.):
            mask = mjj> 0.
            thresh = 0.
        else:
            
            print("bins: %.0f %.0f %.0f %.0f" % (sb_mlow, mjj_low, mjj_high, sb_mhigh))
            low_sb = ((mjj> sb_mlow) & (mjj < mjj_low))
            high_sb =  ((mjj > mjj_high) & (mjj < sb_mhigh))

            in_sb = low_sb | high_sb
            if('sideband' in eff_definition):
                print("Using sideband definition of efficiency")
                if("weighted" not in eff_definition):
                    thresh = np.percentile(jj_scores[in_sb], percentile_cut)
                else:
                    n_low_sb = np.sum(low_sb)
                    n_high_sb = np.sum(high_sb)
                    sb_weights = np.zeros_like(jj_scores)
                    sb_weights[low_sb] = 1.
                    sb_weights[high_sb] = n_low_sb / n_high_sb
                    print("nlow sb %.0f nhigh sb %.0f, weight = %.2f" % (n_low_sb, n_high_sb, n_high_sb / n_low_sb))

                    quantile = percentile_cut / 100. #Uses 0-1 scale instead of 0-100
                    thresh = weighted_quantile(jj_scores[in_sb].reshape(-1), [quantile], sample_weight = sb_weights[in_sb].reshape(-1))[0]
                    old_thresh = np.percentile(jj_scores[in_sb], percentile_cut)

                    new_eff = np.mean(jj_scores > thresh)
                    old_eff = np.mean(jj_scores > old_thresh)

                    print("New thresh %.4f (eff %.4f), old thresh %.4f (%.4f)" % (thresh, new_eff, old_thresh, old_eff))

                    
            elif('central' in eff_definition):
                sig_region_mjj = (mjj_high - mjj_low ) / 2.0
                central_bin_low = 3400.
                central_bin_high = 3600.
                if(options.mbin == 4):
                    central_bin_low = 3700.
                    central_bin_high = 4000.
                elif(options.mbin == 14 or (sig_region_mjj > central_bin_low and sig_region_mjj < central_bin_high) ):
                    central_bin_low = 3000.
                    central_bin_high = 3250.

                print("Using efficiency defined in central region %.0f-%0.f" % (central_bin_low, central_bin_high))
                central_mask = (mjj > central_bin_low) & (mjj  < central_bin_high)
                thresh = np.percentile(jj_scores[central_mask], percentile_cut)
                eff = np.mean(jj_scores > thresh)
                print("Overall eff is %.4f" % eff)

            else:
                thresh = np.percentile(jj_scores, percentile_cut)

            mask = jj_scores > thresh


        mjj_output = mjj[mask]
        is_sig_output = Y[mask]
        event_num_output = event_num[mask]
        print("Selected %i events" % mjj_output.shape[0])
        eps = 1e-6

        in_window_all = (mjj > options.mjj_low) & (mjj < options.mjj_high)
        in_window = (mjj_output > options.mjj_low) & (mjj_output < options.mjj_high)
        sig_events = is_sig_output > 0.9
        bkg_events = is_sig_output < 0.1
        S = mjj_output[sig_events & in_window].shape[0] + eps
        B = mjj_output[bkg_events & in_window].shape[0] + eps

        

        nsig = (Y[Y > 0.9]).shape[0] 
        nbkg = (Y[Y < 0.1]).shape[0] 

        
        sig_eff_deta = 1.0 #need separate file to compute properly

        if(nsig > 0):
            sig_eff = float(( Y[(Y > 0.9) & mask]).shape[0]) / nsig
            sig_eff_window = S / nsig
        else:
            sig_eff = sig_eff_window = -1.

        bkg_eff = float(( Y[(Y < 0.1) & mask]).shape[0]) / nbkg
        minor_bkg_eff = float(( Y [(Y< -0.1) & mask]).shape[0]) / Y[ (Y < -0.1)].shape[0]
        bkg_eff_window =  B/ Y[(Y< 0.1) & in_window_all].shape[0]

        sig_eff_window_nosel = -1.0


        print("Mjj window %f to %f " % (options.mjj_low, options.mjj_high))
        print("S/B %f, sigificance ~ %.1f " % (float(S)/B, S/np.sqrt(B)))
        print("Sig Eff %.3f, with window %.3f " % (sig_eff, sig_eff_window))
        print("Bkg eff %.3f, in mjj window %.3f " % (bkg_eff, bkg_eff_window))
        print("Minor bkg eff %.3f" % minor_bkg_eff)

        if(sig_only_data is not None):


            print("Computing signal efficiency on signal only file")

            #signal weights
            sig_weights = sig_only_data['sys_weights'][:,0]

            if(len(options.sig_sys) > 0 and options.sig_sys in sys_weights_map.keys()):
                print("Using systematic %s" % options.sig_sys)
                weight_idx = sys_weights_map[options.sig_sys]
                sig_weights *= sig_only_data['sys_weights'][:,weight_idx]

            no_cut_weight = np.sum(sig_weights)

            #deta eff
            sig_deta = sig_only_data['jet_kinematics'][:,1]
            sig_deta_mask = sig_deta >= options.deta_min
            if(options.deta > 0.): sig_deta_mask = sig_deta_mask & (sig_deta < options.deta)

            sig_deta_eff = np.sum(sig_weights[sig_deta_mask]) / no_cut_weight



            sig_eff_deta = np.mean(sig_deta_mask)

            #mjj eff after deta cut
            sig_mjj = sig_only_data['jet_kinematics'][:,0][sig_deta_mask]
            sig_mjj_mask = sig_mjj >= options.mjj_low
            if(options.mjj_high > 0.): sig_mjj_mask = sig_mjj_mask & (sig_mjj < options.mjj_high)

            sig_cut_mask = jj_sig_scores[sig_deta_mask] > thresh

            sig_eff = np.sum(sig_weights[sig_deta_mask][sig_cut_mask]) / np.sum(sig_weights[sig_deta_mask])
            sig_eff_window = np.sum(sig_weights[sig_deta_mask][sig_cut_mask & sig_mjj_mask]) / np.sum(sig_weights[sig_deta_mask])
            sig_eff_window_nosel = np.sum(sig_weights[sig_deta_mask][sig_mjj_mask]) / np.sum(sig_weights[sig_deta_mask])



            sys_effs = dict()
            for sys in sys_weights_map.keys():
                if(sys == 'nom_weight'): continue
                sys_idx = sys_weights_map[sys]
                weights = sig_only_data['sys_weights'][:,0] * sig_only_data['sys_weights'][:,sys_idx]
                sys_sig_eff = np.sum(weights[sig_deta_mask][sig_cut_mask])/ np.sum(weights[sig_deta_mask])
                sys_sig_eff_window = np.sum(weights[sig_deta_mask][sig_cut_mask & sig_mjj_mask])/ np.sum(weights[sig_deta_mask])
                sys_effs[sys]= (sys_sig_eff, sys_sig_eff_window)


            print("Sig eta eff %.3f, sig_eff %.3f, sig_eff_window %.3f sig_eff_window_nosel %.3f" % (sig_eff_deta, sig_eff, sig_eff_window, sig_eff_window_nosel))

            sig_output_mjj = sig_mjj[sig_cut_mask]
            print("mean input sig mass", np.mean(sig_only_data['jet_kinematics'][:,0][sig_deta_mask]))
            print("mean output sig mass", np.mean(sig_output_mjj))
            print(sig_output_mjj[0])
            sig_output_weights = sig_only_data['sys_weights'][:,0][sig_deta_mask][sig_cut_mask]

            if('fit_inputs' in output_name):
                sig_output_name = output_name.replace('fit_inputs', 'sig_shape')
            else:
                sig_output_name = output_name.replace('.h5', '_sig_shape.h5')

            print("Creating %s" % sig_output_name)
            with h5py.File(sig_output_name, "w") as f_sig:
                f_sig.create_dataset('mjj', data = sig_output_mjj, chunks = True, maxshape = (None))
                f_sig.create_dataset('weights', data = sig_output_weights, chunks = True, maxshape = (None))
                f_sig.create_dataset('truth_label', data = np.ones((sig_output_mjj.shape[0],1)), chunks = True, maxshape = (None, 1))
                #print(f_sig['mjj'][0], np.mean(f_sig['mjj']))


        print("Outputting to %s \n\n" % output_name)


        with h5py.File(output_name, "w") as f:
            
            mjj_shape = list(mjj_output.shape)
            is_sig_shape = list(is_sig_output.shape)
            event_num_shape = list(event_num_output.shape)

            mjj_shape[0] = None
            is_sig_shape[0] = None
            event_num_shape[0] = None
            f.create_dataset("sig_eff", data=np.array([sig_eff]) )
            f.create_dataset("sig_eff_deta", data=np.array([sig_eff_deta]) )
            f.create_dataset("sig_eff_window", data=np.array([sig_eff_window]) )
            f.create_dataset("sig_eff_window_nosel", data=np.array([sig_eff_window_nosel]) )
            f.create_dataset("score_thresh", data = np.array([thresh]))
            if(sig_only_data is not None):
                for sys in sys_weights_map.keys():
                    if(sys == 'nom_weight'): continue
                    print(sys, sys_effs[sys])
                    f.create_dataset("sig_eff" + "_"+ sys, data=np.array([sys_effs[sys][0]]) )
                    f.create_dataset("sig_eff_window" +"_" + sys , data=np.array([sys_effs[sys][1]]) )


            if(not options.eff_only):
                f.create_dataset("mjj", data=mjj_output, chunks = True, maxshape = mjj_shape)
                f.create_dataset("truth_label", data=is_sig_output, chunks = True, maxshape = is_sig_shape)
                f.create_dataset("event_num", data=event_num_output, chunks = True, maxshape = event_num_shape)


        if(options.do_roc):
            #sig_eff = np.clip(sig_eff, 1e-8, 1.)
            #bkg_eff = np.clip(bkg_eff, 1e-8, 1.)
            f_np = output_name.replace(".h5", "_effs.npz")
            print("Creating %s" % f_np)

            np.savez(f_np, sig_eff = sig_eff_roc, bkg_eff = bkg_eff_roc,
                    j1_quantiles = j1_qs[in_window_and_nominor_bkg], j2_quantiles = j2_qs[in_window_and_nominor_bkg], Y = Y[in_window_and_nominor_bkg])

        make_mjj_eff_plot = True
        if(make_mjj_eff_plot):
            f_plt = output_name.replace(".h5", "_mjj_eff_plot.png")
            print("Creating %s" % f_plt)
            n_bins = 20
            ratio_range = [0.0, 0.1]
            make_ratio_histogram([mjj_output, mjj], ["Selected", "Inclusive"], 
                ['blue', 'green'], "Mjj", "", n_bins, ratio_range = ratio_range, weights = None, errors = True, 
                            normalize=False, save = True, fname=f_plt, logy = True, max_rw = -1)






    del data



if(__name__ == "__main__"):
    if(len(sys.argv) ==2): #use a dict of parameters
        fname = sys.argv[1]
        options = get_options_from_json(fname)


    else:
        parser = input_options()
        parser.add_argument("--effs", nargs="+", default = [], type = float)
        parser.add_argument("--do_roc", default = False, help = "Save info for roc")
        options = parser.parse_args()

    classifier_selection(options)
