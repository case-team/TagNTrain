import sys
sys.path.append('..')
from utils.TrainingUtils import *
from optparse import OptionParser
from optparse import OptionGroup
import random

import h5py



#################################################################




def tag_and_train(options):
    if(options.use_images): network_type = "CNN"
    else: network_type = "dense"

#which images to train on and which to use for labelling
    if(options.training_j == 1):
        j_label = "j1_"
        opp_j_label = "j2_"
        print("training classifier for j1 using j2 for labeling")

    elif (options.training_j ==2):
        j_label = "j2_"
        opp_j_label = "j1_"
        print("training classifier for j2 using j1 for labeling")
    else:
        print("Training jet not 1 or 2! Exiting")
        exit(1)

    if(options.labeler_name == ""):
        print("Must provide labeler name!")
        exit(1)





    plot_prefix = "TNT" + str(options.tnt_iter) + "_" + network_type 

#start with different data not to overlap training sets
    data_start = options.data_start
    print("TNT iter is ", options.tnt_iter)




    options.keep_low = options.keep_high = -1.

    if(not options.no_mjj_cut):
        #keep window size proportional to mjj bin center
        window_size = (options.mjj_high - options.mjj_low)/2.
        window_frac = window_size / ((options.mjj_high + options.mjj_low)/ 2.)

        window_low_size = window_frac*options.mjj_low / (1 + window_frac)
        window_high_size = window_frac*options.mjj_high / (1 - window_frac)
        options.keep_low = options.mjj_low - window_low_size
        options.keep_high = options.mjj_high + window_high_size

        print("Mjj keep low %.0f keep high %.0f \n" % ( options.keep_low, options.keep_high))



    options.keys = []
    l_key2 = ""
    if(options.use_images):
        options.keys = ['mjj', 'j1_images', 'j2_images']
        if(not options.no_ptrw): options.keys.append('jet_kinematics')
        x_key = j_label +  'images'
        l_key = opp_j_label +  'images'
    else:
        options.keys = ['mjj', 'j1_features', 'j2_features']
        if(not options.no_ptrw): options.keys.append('jet_kinematics')
        x_key = j_label +  'features'
        if('auto' in options.labeler_name):
            options.keys.append(opp_j_label + "images")
            l_key = opp_j_label + 'images'
            if(options.randsort):
                options.keys.append(j_label + "images")
                l_key2 = j_label + "images"
                x_key2 = opp_j_label +  'features'
        else:
            l_key = opp_j_label +  'features'
            if(options.randsort): 
                l_key2 = j_label +  'features'
                x_key2 = opp_j_label +  'features'

    print("Keys: ", options.keys)
    import time
    t1 = time.time()

#load the dataset

    t1 = time.time()
    data, val_data = load_dataset_from_options(options)
    

    data2 = val_data2 = None
    if(options.randsort):
        data2 = copy.deepcopy(data)
        val_data2 = copy.deepcopy(val_data)
    do_val = val_data is not None

    t2 = time.time()
    print("load time  %s " % (t2 -t1))





#labeler_plot = plot_dir+ opp_j_label +  plot_prefix + "_labeler_regions.png"
#pt_plot = plot_dir + j_label + plot_prefix + "pt_dists.png"
#pt_rw_plot = plot_dir + j_label + plot_prefix + "pt_rw_dists.png"


    print("\n Loading labeling model from %s \n" % options.labeler_name)
    labeler = tf.keras.models.load_model(options.labeler_name)

    labeler_scores = data.labeler_scores(labeler,  l_key)


    print("Sig-rich region defined > %i percentile" %options.sig_cut)
    print("Bkg-rich region defined < %i percentile" %options.bkg_cut)

    sig_region_cut = np.percentile(labeler_scores, options.sig_cut)
    bkg_region_cut = np.percentile(labeler_scores, options.bkg_cut)

    print("Labeler: cut high %.3e, cut low %.3e " % (sig_region_cut, bkg_region_cut))

    data.make_Y_TNT(sig_region_cut = sig_region_cut, bkg_region_cut = bkg_region_cut, cut_var = labeler_scores, mjj_low = options.mjj_low, mjj_high = options.mjj_high)
    print_signal_fractions(data['label'], data['Y_TNT'])
    #options.batch_size *= 1./((sig_region_cut + bkg_region_cut)/200)

    if(options.randsort):
        labeler_scores2 = data2.labeler_scores(labeler, l_key2)
        sig_region_cut2 = np.percentile(labeler_scores2, options.sig_cut)
        bkg_region_cut2 = np.percentile(labeler_scores2, options.bkg_cut)
        print("Labeler2: cut high %.3e, cut low %.3e " % (sig_region_cut2, bkg_region_cut2))
        j1_bins, j1_ratio = make_ratio_histogram([labeler_scores, labeler_scores2], ["J1 scores", "J2 scores"], ["r", "b"], 'Score', "", 30,
                        normalize=False, weights = None, save = True, fname="jrand_labeler_scores_cmp.png")


        data2.make_Y_TNT(sig_region_cut = sig_region_cut2, bkg_region_cut = bkg_region_cut2, cut_var = labeler_scores2, 
                         mjj_low = options.mjj_low, mjj_high = options.mjj_high, extra_str = '2')
        print_signal_fractions(data2['label'], data2['Y_TNT2'])

    if(do_val):
        val_labeler_scores = val_data.labeler_scores(labeler,  l_key)
        val_data.make_Y_TNT(sig_region_cut = sig_region_cut, bkg_region_cut = bkg_region_cut, cut_var = val_labeler_scores, mjj_low = options.mjj_low, mjj_high = options.mjj_high)
        if(options.randsort):
            val_labeler_scores2 = val_data2.labeler_scores(labeler, l_key2)
            val_data2.make_Y_TNT(sig_region_cut = sig_region_cut2, bkg_region_cut = bkg_region_cut2, cut_var = val_labeler_scores2, 
                                 mjj_low = options.mjj_low, mjj_high = options.mjj_high, extra_str = '2')



    sample_weights = sample_weights2 = None
    if(not options.no_sample_weights):
        sample_weights = sample_weights2 = "weight"
    if(not options.no_ptrw): 
        sample_weights = j_label + 'ptrw'

        data.make_ptrw('Y_TNT', save_plots = False)
        if(options.randsort): 
            data2.make_ptrw('Y_TNT2', save_plots = False, extra_str = '2')
            sample_weights2 = opp_j_label + 'ptrw2'
        if(do_val): 
            val_data.make_ptrw('Y_TNT', save_plots = False)
            if(options.randsort): val_data2.make_ptrw('Y_TNT2', save_plots = False, extra_str = '2')





    myoptimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.8, beta_2=0.99, epsilon=1e-08, decay=0.0005)

    cbs = [tf.keras.callbacks.History()]
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-6, patience=10, verbose=1, mode='min')
    cbs.append(early_stop)



    # train model
    t_data = data.gen(x_key,'Y_TNT', key3 = sample_weights,  batch_size = options.batch_size)
    if(options.randsort):
        t_data.add_dataset(x_key2, 'Y_TNT2', key3 = sample_weights2, dataset = data2)

    v_data = None
    n_val = 0
    if(do_val): 
        #v_data = val_data.gen(x_key,'label', key3 = sample_weights, batch_size = options.batch_size) #true labels
        v_data = val_data.gen(x_key,'Y_TNT', key3 = sample_weights, batch_size = options.batch_size) #TNT labels
        val_data_plain = [val_data[x_key], val_data['label'], val_data['Y_TNT'], val_data[sample_weights]]
        if(options.randsort):
            v_data.add_dataset(x_key2, 'Y_TNT2', key3 = sample_weights + '2', dataset = val_data2)
            print(val_data_plain[0].shape, val_data_plain[1].shape)
            val_data_plain[0] = np.append(val_data_plain[0], val_data2[x_key2], axis =0)
            val_data_plain[1] = np.append(val_data_plain[1], val_data2['label'], axis = 0)
            val_data_plain[2] = np.append(val_data_plain[2], val_data2['Y_TNT2'], axis = 0)
            val_data_plain[3] = np.append(val_data_plain[3], val_data2[sample_weights + '2'], axis = 0)
            print(val_data_plain[0].shape, val_data_plain[1].shape)
        roc = RocCallback(training_data=(np.zeros(100), np.zeros(100)), validation_data=val_data_plain[:2], extra_label = "true: ")
        cbs.append(roc)
        n_val = v_data.nTotal

    print("Will train on %i events, validate on %i events" % (t_data.nTotal, n_val))



    for seed in options.seeds:
        np.random.seed(seed)
        tf.set_random_seed(seed)
        os.environ['PYTHONHASHSEED']=str(seed)
        random.seed(seed)

        cnn_shape = (32,32,1)
        dense_shape = 8
        model_list = []

        for model_idx in range(options.num_models):
            print("Creating model %i" % model_idx)
            if(options.model_start == ""):
                print("Creating new model based on seed %i " % seed)
                if(options.use_images):
                    model = CNN(cnn_shape)
                else:
                    model = dense_net(dense_shape)

                model.summary()
                model.compile(optimizer=myoptimizer,loss='binary_crossentropy', metrics = ['accuracy']
                        )
            else:
                print("Starting with model from %s " % options.model_start)
                model = tf.keras.models.load_model(options.model_dir + j_label + options.model_start)



            #additional_val = AdditionalValidationSets([(X_val, Y_true_val, "Val_true_sig")], options.batch_size = 500)

            #cbs = [callbacks.History(), additional_val, roc1, roc2] 



            history = model.fit(t_data, 
                    epochs = options.num_epoch, 
                    validation_data = v_data,
                    callbacks = cbs,
                    verbose = 2 )
            model_list.append(model)

            preds = model.predict_proba(data[x_key][:10])
            if np.any(np.isnan(preds)): 
                print("Got output Nans. Should rerun with a different seed")
                sys.exit(1)


        if(options.num_models == 1):
            best_model = model_list[0]
        else:

            min_loss = 9999999
            best_i = -1

            val_sig_events = val_data_plain[2] > 0.9
            val_bkg_events = val_data_plain[2] < 0.1
            for model_idx in range(options.num_models):
                preds = model_list[model_idx].predict(val_data_plain[0])
                loss = bce(preds.reshape(-1), val_data_plain[2].reshape(-1), weights = val_data_plain[3])


                true_loss = auc = -1
                if(np.sum(val_data_plain[1]) > 10):
                    true_loss = bce(preds.reshape(-1), val_data_plain[1].reshape(-1))
                    auc = roc_auc_score(val_data_plain[1], preds)
                eff_cut_metric = compute_effcut_metric(preds[val_sig_events], preds[val_bkg_events], eff = 0.01)
                print("Model %i,  loss %.3f, true loss %.3f, auc %.3f, effcut metric %.3f" % (model_idx, loss, true_loss, auc, eff_cut_metric))
                loss = -eff_cut_metric
                if(loss < min_loss):
                    min_loss = loss
                    best_i = model_idx
            print("Selecting model %i " % best_i)
            best_model = model_list[best_i]





        if(do_val and np.sum(val_data_plain[1]) > 10):
            msg = "End of training. "
            y_pred_val = best_model.predict_proba(val_data_plain[0])
            roc_val = roc_auc_score(val_data_plain[1], y_pred_val)
            phrase = " roc-auc_val: %s (based on %i signal validation events)" % (str(round(roc_val,4)), np.sum(val_data_plain[1]))
            msg += phrase
            print(msg, end =100*' ' + '\n')

        f_model = options.output
        if('{seed}' in options.output):
            f_model = f_model.format(seed = seed)
        print("Saving model to : ", f_model)
        best_model.save(f_model)

        #training_plot = options.plot_dir + j_label + plot_prefix + "training_history.png"
        #plot_training(history.history, fname = training_plot)
    del data
    if(data2 is not None): del data2
    if(val_data is not None): del val_data
    if(val_data2 is not None): del val_data2


if(__name__ == "__main__"):
    parser = input_options()
    options = parser.parse_args()
    tag_and_train(options)
