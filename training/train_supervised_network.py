import sys
sys.path.append('..')
from utils.TrainingUtils import *
from optparse import OptionParser
from optparse import OptionGroup
import random
import time



def train_supervised_network(options):
    compute_mjj_window(options)

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
    print("Enforcing no minor bkgs for supervised training")
    options.no_minor_bkgs = True
    data, val_data = load_dataset_from_options(options)
    do_val = val_data is not None
    t2 = time.time()
    print("load time  %s " % (t2 -t1))


    mjjs  = data['mjj'][()]
    sig_events = data['label'].reshape(-1) > 0.9

    if(options.preprocess != ""):
        print("\n Doing preprocess %s \n" % options.preprocess)
        feats = data[x_key]
        qts = create_transforms(feats, dist = options.preprocess)
        data.make_preprocessed_feats(x_key, qts)
        if(do_val): val_data.make_preprocessed_feats(x_key, qts)
        x_key = x_key + "_normed"



    #print(mjjs[sig_events] [:10])


    t_data = data.gen(x_key,'label', key3 = None, batch_size = options.batch_size)
    v_data = None

    cbs = [tf.keras.callbacks.History()]
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=5 + options.num_epoch/20, verbose=1, mode='min')
    cbs.append(early_stop)

    nVal = 0
    if(do_val): 
        nVal = val_data.nTrain
        v_data = val_data.gen(x_key,'label', key3 = None, batch_size = options.batch_size) 

        roc = RocCallback(training_data=(np.zeros(100), np.zeros(100)), validation_data=(val_data[x_key], val_data['label']), extra_label = "true: ")
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
            if(not options.large_net):
                model = CNN(cnn_shape)
            else:
                model = CNN_large(cnn_shape)
        else:
            dense_shape = data[x_key].shape[-1]
            if(options.small_net):
                model = dense_small_net(dense_shape)
            else:
                model = dense_net(dense_shape)

        model.compile(optimizer=myoptimizer,loss='binary_crossentropy', metrics = ['accuracy'])
        if(model_idx == 0): model.summary()

        history = model.fit(t_data, 
                epochs = options.num_epoch, 
                validation_data = v_data,
                callbacks = cbs,
                verbose = 2 )
        model_list.append(model)

    if(options.num_models == 1):
        best_model = model_list[0]
    else:
        min_loss = 9999999
        best_i = -1

        for model_idx in range(options.num_models):
            preds = model_list[model_idx].predict(val_data[x_key])
            true_loss = bce(preds.reshape(-1), np.clip(val_data['label'], 0, 1 ).reshape(-1))
            auc = roc_auc_score(np.clip(val_data['label'], 0, 1), preds)
            print("Model %i, true loss %.3f, auc %.3f " % (model_idx, true_loss, auc))
            loss = true_loss
            if(loss < min_loss):
                min_loss = loss
                best_i = model_idx
        print("Selecting model %i " % best_i)
        best_model = model_list[best_i]


    if(do_val and np.sum(val_data['label'] > 0) > 10):
        msg = "End of training. "
        y_pred_val = best_model.predict_proba(val_data[x_key])
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
    train_supervised_network(options)


