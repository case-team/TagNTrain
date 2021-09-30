import sys
sys.path.append('..')
from utils.TrainingUtils import *
import h5py

    


def classifier_selection(options):
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
        else:
            jj_scores = get_jj_scores("", f, options.model_type, jj_images, jj_dense_inputs, num_models = options.num_models)


    for eff in options.effs:
        print("Will select events with efficiency %.3f" % eff)
        percentile_cut = 100. - eff
        if('{eff}' in options.output):
            output_name = options.output.format(eff = eff)
        else:
            output_name = options.output

        if(eff == 100.):
            mask = mjj> 0.
        elif(options.model_type <3):
            mask = make_selection(j1_score, j2_score, percentile_cut)
        else:
            thresh = np.percentile(jj_scores, percentile_cut)
            mask = scores > thresh

        mjj_output = mjj[mask]
        is_sig_output = Y[mask]
        event_num_output = event_num[mask]
        print("Selected %i events" % mjj_output.shape[0])

        in_window = (mjj_output > options.mjj_low) & (mjj_output < options.mjj_high)
        sig_events = is_sig_output > 0.9
        bkg_events = is_sig_output < 0.1
        S = mjj_output[sig_events & in_window].shape[0]
        B = mjj_output[bkg_events & in_window].shape[0]
        print("Mjj window %f to %f " % (options.mjj_low, options.mjj_high))
        print("S/B %f, sigificance ~ %.1f " % (float(S)/B, S/np.sqrt(B)))
        print("Outputting to %s \n\n" % output_name)


        with h5py.File(output_name, "w") as f:
            
            mjj_shape = list(mjj_output.shape)
            print(mjj_shape)
            is_sig_shape = list(is_sig_output.shape)
            event_num_shape = list(event_num_output.shape)

            mjj_shape[0] = None
            is_sig_shape[0] = None
            event_num_shape[0] = None
            f.create_dataset("mjj", data=mjj_output, chunks = True, maxshape = mjj_shape)
            f.create_dataset("truth_label", data=is_sig_output, chunks = True, maxshape = is_sig_shape)
            f.create_dataset("event_num", data=event_num_output, chunks = True, maxshape = event_num_shape)




    del data



if(__name__ == "__main__"):
    parser = input_options()
    parser.add_argument("--effs", nargs="+", default = [], type = float)
    options = parser.parse_args()

    classifier_selection(options)
