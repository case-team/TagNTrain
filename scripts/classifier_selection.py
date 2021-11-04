import sys
sys.path.append('..')
from utils.TrainingUtils import *
import h5py

    


def classifier_selection(options):
    #model types: 0 CNN (one jet), 1 auto encoder, 2 dense (one jet), 3 CNN (both jets), 4 dense (both jets), 5 is VAE 
    if(len(options.effs) == 0):
        print("Must input at least 1 efficiency value!")




    n_points = 200.




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

    Y = data['label'].reshape(-1)
    mjj = data['mjj']

    event_num = data['event_info'][:,0]

    j1_images = j2_images = jj_images = j1_dense_inputs = j2_dense_inputs = jj_dense_inputs = None

    if('j1_images' in options.keys): 
        j1_images = data['j1_images']
        j2_images = data['j2_images']
    if('jj_images' in options.keys): jj_images = data['jj_images']
    if('j1_features' in options.keys):
        j1_dense_inputs = data['j1_features']
        j2_dense_inputs = data['j2_features']

    if('jj_dense_inputs' in options.keys):
        jj_dense_inputs = data['jj_features']
    batch_size = 1024
    sig_effs = []
    bkg_effs = []
    overall_effs = []

    if(len(options.effs) ==1 and options.effs[0] == 100.):
        scores = None

    else:
        if('sig_idx' in options.labeler_name): 
            f = options.labeler_name.format(sig_idx = options.sig_idx)
        else:
            f = options.labeler_name

        print("Using model %s" % f)
        if(options.model_type <= 2 or options.model_type == 5): #classifier on each jet

            #j1_score, j2_score = get_jet_scores(model_dir, f, options.model_type, j1rand_images, j2rand_images, j1rand_dense_inputs, j2rand_dense_inputs, num_models = options.num_models)
            j1_score, j2_score = get_jet_scores("", f, options.model_type, j1_images, j2_images, j1_dense_inputs, j2_dense_inputs, num_models = options.num_models)

            if(options.do_roc):
                in_window = (mjj > options.mjj_low) & (mjj < options.mjj_high)
                j1_qs = quantile_transform(j1_score[in_window].reshape(-1,1), copy = True).reshape(-1)
                j2_qs = quantile_transform(j2_score[in_window].reshape(-1,1), copy = True).reshape(-1)
                Y_inwindow = Y[in_window]
                #sig_effs = np.array([(Y_inwindow[(j1_qs > perc) & (j2_qs > perc) & (Y_inwindow==1)].shape[0])/(Y_inwindow[Y_nwindow==1].shape[0]) for perc in np.arange(0.,1., 1./n_points)])
                #bkg_effs = np.array([(Y_inwindow[(j1_qs > perc) & (j2_qs > perc) & (Y_inwindow==0)].shape[0])/(Y_inwindow[Y_inwindow==0].shape[0]) for perc in np.arange(0.,1., 1./n_points)])
                for perc in np.arange(0., 1., 1./n_points):
                    mask = (j1_qs > perc) &  (j2_qs > perc)
                    eff = np.mean(mask)
                    sig_eff = np.mean(mask & (Y_inwindow ==1)) / np.mean(Y_inwindow == 1)
                    bkg_eff = np.mean(mask & (Y_inwindow ==0)) / np.mean(Y_inwindow == 0)

                    sig_effs.append(sig_eff)
                    bkg_effs.append(bkg_eff)
                    overall_effs.append(eff)

        else:
            jj_scores = get_jj_scores("", f, options.model_type, jj_images, jj_dense_inputs, num_models = options.num_models)
            if(options.do_roc):
                in_window = (mjj > options.mjj_low) & (mjj < options.mjj_high)
                bkg_eff, sig_eff, thresholds_cwola = roc_curve(Y[in_window], jj_scores[in_window])



    for eff in options.effs:

        use_sidebands = True
        window_size = (options.mjj_high - options.mjj_low)/2.
        window_frac = window_size / ((options.mjj_high + options.mjj_low)/ 2.)

        window_low_size = window_frac*options.mjj_low / (1 + window_frac)
        window_high_size = window_frac*options.mjj_high / (1 - window_frac)
        sb_mlow = options.mjj_low - window_low_size
        sb_mhigh = options.mjj_high + window_high_size

        in_sb = ((mjj> sb_mlow) & (mjj < options.mjj_low)) | ((mjj > options.mjj_high) & (mjj < sb_mhigh))

        print("Will select events with efficiency %.3f" % eff)
        percentile_cut = 100. - eff
        if('{eff}' in options.output):
            output_name = options.output.format(eff = eff)
        else:
            output_name = options.output

        if(eff == 100.):
            mask = mjj> 0.
        elif(options.model_type <3):
            if(use_sidebands):
                mask = make_selection(j1_score, j2_score, percentile_cut, mask = in_sb)
            else:
                mask = make_selection(j1_score, j2_score, percentile_cut, mask = None)

        else:
            if(use_sidebands):
                thresh = np.percentile(jj_scores[in_sb], percentile_cut)
            else:
                thresh = np.percentile(jj_scores, percentile_cut)

            mask = scores > thresh



        mjj_output = mjj[mask]
        is_sig_output = Y[mask]
        event_num_output = event_num[mask]
        print("Selected %i events" % mjj_output.shape[0])

        in_window_all = (mjj > options.mjj_low) & (mjj < options.mjj_high)
        in_window = (mjj_output > options.mjj_low) & (mjj_output < options.mjj_high)
        sig_events = is_sig_output > 0.9
        bkg_events = is_sig_output < 0.1
        S = mjj_output[sig_events & in_window].shape[0]
        B = mjj_output[bkg_events & in_window].shape[0]

        nsig = (Y[Y > 0.9]).shape[0] 
        nbkg = (Y[Y < 0.1]).shape[0] 

        if(nsig > 0):
            sig_eff = float(( Y[(Y > 0.9) & mask]).shape[0]) / nsig
            sig_eff_window = S / nsig
        else:
            sig_eff = sig_eff_window = -1.

        bkg_eff = float(( Y[(Y < 0.1) & mask]).shape[0]) / nbkg
        minor_bkg_eff = float(( Y [(Y< -0.1) & mask]).shape[0]) / Y[ (Y < -0.1)].shape[0]
        bkg_eff_window =  B/ Y[(Y< 0.1) & in_window_all].shape[0]


        print("Mjj window %f to %f " % (options.mjj_low, options.mjj_high))
        print("S/B %f, sigificance ~ %.1f " % (float(S)/B, S/np.sqrt(B)))
        print("Sig Eff %.3f, with window %.3f " % (sig_eff, sig_eff_window))
        print("Bkg eff %.3f, in mjj window %.3f " % (bkg_eff, bkg_eff_window))
        print("Minor bkg eff %.3f" % minor_bkg_eff)
        print("Outputting to %s \n\n" % output_name)


        with h5py.File(output_name, "w") as f:
            
            mjj_shape = list(mjj_output.shape)
            is_sig_shape = list(is_sig_output.shape)
            event_num_shape = list(event_num_output.shape)

            mjj_shape[0] = None
            is_sig_shape[0] = None
            event_num_shape[0] = None
            f.create_dataset("mjj", data=mjj_output, chunks = True, maxshape = mjj_shape)
            f.create_dataset("truth_label", data=is_sig_output, chunks = True, maxshape = is_sig_shape)
            f.create_dataset("event_num", data=event_num_output, chunks = True, maxshape = event_num_shape)
            f.create_dataset("sig_eff", data=np.array([sig_eff]) )
            f.create_dataset("sig_eff_window", data=np.array([sig_eff_window]) )


        if(options.do_roc):
            #sig_eff = np.clip(sig_eff, 1e-8, 1.)
            #bkg_eff = np.clip(bkg_eff, 1e-8, 1.)
            f_np = options.output.replace(".h5", "_effs.npz")

            np.savez(f_np, sig_eff = sig_effs, bkg_eff = bkg_effs, overall_eff = overall_effs)

        make_mjj_eff_plot = True
        if(make_mjj_eff_plot):
            f_plt = options.output.replace(".h5", "_mjj_eff_plot.png")
            print("Creating %s" % f_plt)
            n_bins = 20
            ratio_range = [0.0, 0.03]
            make_ratio_histogram([mjj_output, mjj], ["Selected", "Inclusive"], 
                ['blue', 'green'], "Mjj", "", n_bins, ratio_range = ratio_range, weights = None, errors = True, 
                            normalize=False, save = True, fname=f_plt, logy = True, max_rw = -1)






    del data



if(__name__ == "__main__"):
    if(len(sys.argv) ==2): #use a dict of parameters
        fname = sys.argv[1]
        options = get_options_from_pkl(fname)


    else:
        parser = input_options()
        parser.add_argument("--effs", nargs="+", default = [], type = float)
        parser.add_argument("--do_roc", default = False, help = "Save info for roc")
        options = parser.parse_args()

    classifier_selection(options)
