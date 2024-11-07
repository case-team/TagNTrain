import sys
sys.path.append('..')
from utils.TrainingUtils import *
from optparse import OptionParser
from optparse import OptionGroup
import random
import time






def train_ttbar_control_region(options):
    print(options.__dict__)

    options.keys = ['mjj', 'j1_features', 'j2_features', 'jet_kinematics']
    if(len(options.sig_file) == 0): 
        options.sig_per_batch = 0
    #options.keep_mlow = 1300.
    #options.keep_mhigh = 1800.
    #options.keep_mhigh = 99999.

    print("Mjj keep low %.0f keep high %.0f \n" % ( options.keep_mlow, options.keep_mhigh))

        

#load the dataset
    t1 = time.time()
    data, val_data = load_dataset_from_options(options)
    do_val = val_data is not None

    data2 = val_data2 = None
    if(options.randsort):
        data2 = copy.deepcopy(data)
        val_data2 = copy.deepcopy(val_data)

    t2 = time.time()
    print("load time  %s " % (t2 -t1))



    Y = (data['label'] == -2)

    #lower pt side is used to 'tag'
    j_label = 'j1_'
    opp_j_label = 'j2_'
    x_key = 'j1_features'
    opp_x_key = 'j2_features'
    filter_frac = data.make_Y_ttbar(data['j2_features'], tau32_cut = options.tau32_cut, deepcsv_cut = options.deepcsv_cut)
    if(options.randsort):
        data2.make_Y_ttbar(data2['j1_features'], tau32_cut = options.tau32_cut, deepcsv_cut = options.deepcsv_cut, extra_str = '2')



    if(do_val):
        val_data.make_Y_ttbar(val_data['j2_features'], tau32_cut = options.tau32_cut, deepcsv_cut = options.deepcsv_cut)
        if(options.randsort):
            val_data2.make_Y_ttbar(val_data2['j1_features'], tau32_cut = options.tau32_cut, deepcsv_cut = options.deepcsv_cut, extra_str = '2')


    #batch_size_scale = 1./filter_frac
    #print("Scaling batch size by %.2f to account for masking" % batch_size_scale)
    #options.batch_size = int(options.batch_size * batch_size_scale)
    #print(options.batch_size)

    


    print_signal_fractions((data['label'] == -2), data['Y_ttbar'])

    sample_weights = None
    if(not options.no_sample_weights):
        sample_weights = "weight"
    if(options.use_one and not options.no_ptrw): 
        sample_weights = j_label + 'ptrw'
        data.make_ptrw('Y_ttbar', use_weights = not options.no_sample_weights, save_plots = False)
        if(options.randsort): 
            data2.make_ptrw('Y_ttbar', use_weights = not options.no_sample_weights, save_plots = False, extra_str = '2')
            sample_weights2 = opp_j_label + 'ptrw2'
        if(do_val): 
            val_data.make_ptrw('Y_ttbar', use_weights = not options.no_sample_weights, save_plots = False)
            if(options.randsort): val_data2.make_ptrw('Y_ttbar', use_weights = not options.no_sample_weights, save_plots = False, extra_str = '2')


    #t_data = data.gen(x_key,'Y_ttbar', key3 = sample_weights, batch_size = options.batch_size)
    x = data[x_key]
    Y = data['Y_ttbar']
    weights = data[sample_weights]
    if(options.supervised): Y = data['label'] == -2
    elif(options.randsort):
        x = np.append(x, data2[opp_x_key], axis = 0)
        Y = np.append(Y, data2['Y_ttbar2'], axis = 0)
        weights = np.append(weights, data2[sample_weights2])


    v_data = None

    cbs = [tf.keras.callbacks.History()]
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-6, patience=5, verbose=1, mode='min')
    cbs.append(early_stop)

    nVal = 0
    if(do_val): 
        nVal = val_data.nTrain
        #v_data = val_data.gen(x_key,'label', key3 = sample_weights, batch_size = options.batch_size) #truth labels
        #v_data = val_data.gen(x_key,'Y_ttbar', key3 = sample_weights, batch_size = options.batch_size) #mjj labels
        v_x = val_data[x_key]
        v_Y = val_data['Y_ttbar']
        v_weights = val_data[sample_weights]
        if(options.supervised): v_Y = val_data['label'] == -2
        elif(options.randsort):
            v_x = np.append(v_x, val_data2[opp_x_key], axis = 0)
            v_Y = np.append(v_Y, val_data2['Y_ttbar2'], axis = 0)
            v_weights = np.append(v_weights, val_data2[sample_weights2])
        v_data = (v_x, v_Y, v_weights)

        #TODO add extra val's to roc callback
        roc = RocCallback(training_data=(np.zeros(100), np.zeros(100)), validation_data=(val_data[x_key], val_data['label'] == -2), extra_label = "true: ")
        cbs.append(roc)




    myoptimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.8, beta_2=0.99, epsilon=1e-08, decay=0.0005)



    print("Will train on %i events, validate on %i events" % (data.nTrain, nVal))
    print(data[x_key][0].shape)
    
    #vary seed for different k-folds
    batch_sum = np.sum(data.batch_list)

    seed = options.seed + batch_sum
    print("Seed is %i" % seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)

    model_list = []

    print("Will train %i models" % options.num_models)
    for model_idx in range(options.num_models):
        print("Creating model %i" % model_idx)

        if(options.use_images):
            if(not options.large):
                model = CNN(cnn_shape)
            else:
                model = CNN_large(cnn_shape)
        else:
            dense_shape = data[x_key].shape[-1]
            model = dense_net(dense_shape)

        model.compile(optimizer=myoptimizer,loss='binary_crossentropy', metrics = ['accuracy'])
        if(model_idx == 0): model.summary()

        #history = model.fit(t_data, 
        #        epochs = options.num_epoch, 
        #        validation_data = v_data,
        #        callbacks = cbs,
        #        verbose = 2 )

        history = model.fit(x = x, y = Y, sample_weight = weights,
                epochs = options.num_epoch,
                validation_data = v_data,
                batch_size = 256,
                callbacks = cbs,
                verbose = 2)
        model_list.append(model)

    Y_val = (val_data['label'] == -2).reshape(-1)
    if(options.num_models == 1):
        best_model = model_list[0]
    else:
        min_loss = 9999999
        best_i = -1

        val_sig_events = val_data['Y_ttbar'] > 0.9
        val_bkg_events = val_data['Y_ttbar'] < 0.1
        for model_idx in range(options.num_models):
            preds = model_list[model_idx].predict(val_data[x_key])
            loss = bce(preds.reshape(-1), val_data['Y_ttbar'][()].reshape(-1), weights = val_data[sample_weights])
            true_loss = auc = -1
            if(np.sum(Y_val)):
                true_loss = bce(preds.reshape(-1), Y_val)
                auc = roc_auc_score(Y_val, preds)
            eff_cut_metric = compute_effcut_metric(preds[val_sig_events], preds[val_bkg_events], eff = 0.10, 
                    weights = val_data[sample_weights][val_bkg_events], labels = Y_val[val_sig_events])
            print("Model %i,  loss %.3f, true loss %.3f, auc %.3f, effcut metric %.3f" % (model_idx, loss, true_loss, auc, eff_cut_metric))
            loss = -eff_cut_metric
            if(loss < min_loss):
                min_loss = loss
                best_i = model_idx
        print("Selecting model %i " % best_i)
        best_model = model_list[best_i]


    if(do_val and np.sum(Y_val) > 10):
        msg = "End of training. "
        y_pred_val = best_model.predict_proba(val_data[x_key])
        roc_val = roc_auc_score(Y_val, y_pred_val)
        phrase = " roc-auc_val: %s (based on %i signal validation events)" % (str(round(roc_val,4)), np.sum(Y_val))
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
    parser.add_argument("--supervised", default = False, action = 'store_true', help = "Supervised")
    parser.add_argument("--tau32_cut", default = 999., type = float, help = "What tau32 cut to use on tag side")
    parser.add_argument("--deepcsv_cut", default = -999., type = float, help = "What deepcsv cut to use on tag side")
    options = parser.parse_args()
    train_ttbar_control_region(options)



