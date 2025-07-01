import sys
sys.path.append('..')
from utils.TrainingUtils import *
from optparse import OptionParser
from optparse import OptionGroup
import random
import time






def train_cwola_hunting_network(options):
    print(options.__dict__)

    compute_mjj_window(options)
    print("Mjj keep low %.0f keep high %.0f \n" % ( options.keep_mlow, options.keep_mhigh))


#which images to train on and which to use for labelling
    if(not options.use_one):
        j_label = "jj_"
        img_key = 'jj_images'
        feat_key = 'jj_features'
        cnn_shape = (32,32,2)
        print("training classifier for both jets")
    elif(options.training_j == 1):
        j_label = "j1_"
        img_key = 'j1_images'
        feat_key = 'j1_features'
        cnn_shape = (32,32,1)
        print("training classifier for j1")

    elif (options.training_j ==2):
        j_label = "j2_"
        img_key = 'j2_images'
        feat_key = 'j2_features'
        cnn_shape = (32,32,1)
        print("training classifier for j2")
    else:
        print("Training jet not 1 or 2! Exiting")
        exit(1)


    options.keys = ['mjj']
    if(options.use_one and not options.no_ptrw):
        options.keys.append('jet_kinematics')

    if(options.use_images):
        options.keys.append(img_key)
        x_key = img_key

    else:
        options.keys.append(feat_key)
        x_key = feat_key
        

#load the dataset
    t1 = time.time()
    data, val_data = load_dataset_from_options(options)
    do_val = val_data is not None
    t2 = time.time()
    print("load time  %s " % (t2 -t1))

    data.make_Y_mjj(options.mjj_low, options.mjj_high)
    if(do_val): val_data.make_Y_mjj(options.mjj_low, options.mjj_high)




    print_signal_fractions(data['label'], data['Y_mjj'])
    mjjs  = data['mjj'][()]
    sig_events = data['label'].reshape(-1) > 0.9

    #print(mjjs[sig_events] [:10])

    sample_weights = None
    if(not options.no_sample_weights):
        sample_weights = "weight"
    if(options.use_one and not options.no_ptrw): 
        sample_weights = j_label + 'ptrw'
        data.make_ptrw('Y_mjj', use_weights = not options.no_sample_weights, save_plots = False)
        #TODO, make validation use same reweighting ratios as training data?
        if(do_val): val_data.make_ptrw('Y_mjj', use_weights = not options.no_sample_weights, save_plots = False)

    if(options.preprocess != ""):
        print("\n Doing preprocess %s \n" % options.preprocess)
        feats = data[x_key]
        qts = create_transforms(feats, dist = options.preprocess)
        data.make_preprocessed_feats(x_key, qts)
        if(do_val): val_data.make_preprocessed_feats(x_key, qts)
        x_key = x_key + "_normed"



    t_data = data.gen(x_key,'Y_mjj', key3 = sample_weights, batch_size = options.batch_size)
    v_data = None


    nVal = 0
    if(do_val): 
        nVal = val_data.nTrain
        #v_data = val_data.gen(x_key,'label', key3 = sample_weights, batch_size = options.batch_size) #truth labels
        v_data = val_data.gen(x_key,'Y_mjj', key3 = sample_weights, batch_size = options.batch_size) #mjj labels





    myoptimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.8, beta_2=0.99, epsilon=1e-08)



    print("Will train on %i events, validate on %i events" % (data.nTrain, nVal))
    print(data[x_key][0].shape)
    
    #vary seed for different k-folds
    batch_sum = np.sum(data.batch_list)

    seed = options.seed + batch_sum
    print("Seed is %i" % seed)
    np.random.seed(seed)
    #tf.set_random_seed(seed)
    #tf.random.set_random_seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)

    model_list = []

    print("Will train %i models" % options.num_models)
    for model_idx in range(options.num_models):
        print("Creating model %i" % model_idx)

        if(options.use_images):
            if(not options.large_net):
                model = CNN(cnn_shape)
            else:
                model = CNN_large(cnn_shape)
        else:
            dense_shape = data[x_key].shape[-1]
            if(options.small_net or data.nTrain < 1e4):
                model = dense_small_net(dense_shape)
            else:
                model = dense_net(dense_shape)

        model.compile(optimizer=myoptimizer,loss='binary_crossentropy', metrics = ['accuracy'])
        if(model_idx == 0): model.summary()



        timeout = TimeOut(t0=time.time(), timeout=30.0/ options.num_models) #stop training after 30 hours to avoid job timeout
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-6, patience=10 + options.num_epoch/20, verbose=1, mode='min')
        checkpoint_loc = "checkpoint.h5"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_loc, monitor = 'val_loss', save_best_only = True, save_weights_only = True)
        cbs = [tf.keras.callbacks.History(), timeout, early_stop, checkpoint]

        if(do_val):
            roc = RocCallback(training_data=(np.zeros(100), np.zeros(100)), validation_data=(val_data[x_key], val_data['label']), extra_label = "true: ")
            cbs.append(roc)

        history = model.fit(t_data, 
                epochs = options.num_epoch, 
                validation_data = v_data,
                callbacks = cbs,
                verbose = 2 )
        model.load_weights(checkpoint_loc)
        model_list.append(model)
        os.system("rm %s" % checkpoint_loc)

    if(options.num_models == 1):
        best_model = model_list[0]
    else:
        min_loss = 9999999
        best_i = -1

        val_sig_events = val_data['Y_mjj'] > 0.9
        val_bkg_events = val_data['Y_mjj'] < 0.1
        if(options.eff_cut * np.sum(val_sig_events)  < 20): options.eff_cut = max(options.eff_cut, 0.1)
        print("Using eff_cut %.2f \n" % options.eff_cut)
        for model_idx in range(options.num_models):
            preds = model_list[model_idx].predict(val_data[x_key])
            loss = bce(preds.reshape(-1), val_data['Y_mjj'][()].reshape(-1), weights = val_data[sample_weights])
            true_loss = auc = -1
            if(np.sum(val_data['label'] > 0) > 10):
                true_loss = bce(preds.reshape(-1), val_data['label'][()].reshape(-1))
                auc = roc_auc_score(np.clip(val_data['label'], 0, 1), preds)
            eff_cut_metric = compute_effcut_metric(preds[val_sig_events], preds[val_bkg_events], eff = options.eff_cut, 
                    weights = val_data[sample_weights][val_bkg_events], labels = val_data['label'][val_sig_events])
            print("Model %i,  loss %.3f, true loss %.3f, auc %.3f, effcut metric %.3f" % (model_idx, loss, true_loss, auc, eff_cut_metric))
            loss = -eff_cut_metric
            if(loss < min_loss):
                min_loss = loss
                best_i = model_idx
        print("Selecting model %i " % best_i)
        best_model = model_list[best_i]


    if(do_val and np.sum(val_data['label'] > 0) > 10):
        msg = "End of training. "
        y_pred_val = best_model.predict(val_data[x_key])
        roc_val = roc_auc_score(np.clip(val_data['label'], 0, 1), y_pred_val)
        phrase = " roc-auc_val: %s (based on %i signal validation events)" % (str(round(roc_val,4)), np.sum(val_data['label'] > 0))
        msg += phrase
        print(msg, end =100*' ' + '\n')

    #plot_training(history.history, options.plot_dir + j_label + "training_history.png")

    f_model = options.output
    if('{seed}' in options.output):
        f_model = f_model.format(seed = seed)
    print("Saving model to : ", f_model)
    best_model.save(f_model)

    del data
    if(val_data is not None): del val_data

if(__name__ == "__main__"):
    parser = input_options()
    options = parser.parse_args()
    train_cwola_hunting_network(options)



