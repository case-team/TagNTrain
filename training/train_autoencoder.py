import sys
sys.path.append('..')
from utils.TrainingUtils import *
from optparse import OptionParser
from optparse import OptionGroup



def train_auto_encoder(options):
    #which images to train on and which to use for labelling
    if(options.training_j == 1):
        j_label = "j1_"
        x_key = 'j1_images'
        opp_j_key = "j2_images"
        print("training autoencoder for j1")

    elif (options.training_j ==2):
        j_label = "j2_"
        x_key = 'j2_images'
        opp_j_key = "j1_images"
        print("training autoencoder for j2")
    else:
        print("Training jet not 1 or 2! Exiting")
        exit(1)

    if(options.num_data == -1):
        data_stop = -1
    else:
        data_stop = options.data_start + options.num_data

    if(not options.no_mjj_cut):
        compute_mjj_window(options)
        print("Requiring mjj window from %.0f to %.0f \n" % (options.keep_mlow, options.keep_mhigh))



    import time
    options.keys  = [x_key, 'mjj']
    if(options.randsort):
        options.keys.append(opp_j_key)

    t1 = time.time()

    data, val_data = load_dataset_from_options(options)
    do_val = val_data is not None

    #restrict to sidebands only
    mjj = data['mjj']
    mjj_cut = (mjj < options.mjj_low) | (mjj > options.mjj_high)
    filter_frac = data.apply_mask(mjj_cut)
    batch_size_scale = 1./filter_frac
    print("Scaling batch size by %.2f to account for masking" % batch_size_scale)
    options.batch_size = int(options.batch_size * batch_size_scale)
    print(options.batch_size)

    t2 = time.time()
    print("load time  %s " % (t2 -t1))



    mjj_cut = data['mjj']


    print("Signal frac is %.3f \n" % (np.mean(data['label'] > 0)))



    cbs = [tf.keras.callbacks.History()]
    myoptimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.8, beta_2=0.99, epsilon=1e-08, decay=0.0005)


# train model
    t_data = data.gen(x_key,x_key, batch_size = options.batch_size)
    if(options.randsort):
        t_data.add_dataset(opp_j_key, opp_j_key, dataset = data)
    v_data = None
    if(do_val): 
        nVal = val_data.nTrain
        v_data = val_data.gen(x_key, x_key, batch_size = options.batch_size)
        if(options.randsort):
            v_data.add_dataset(opp_j_key, opp_j_key, dataset = val_data)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta = 1e-4, verbose=1, mode='min', baseline=None)
        cbs.append(early_stop)
    else:
        nVal = 0

    print("Will train on %i events, validate on %i events" % (data.nTrain, nVal))
    #print(np.mean(data['val_label']))
    #print(np.mean(data['label']))
    


    model_list = []
    histories = []

    print("Will train %i models" % options.num_models)
    for model_idx in range(options.num_models):
        print("Creating model %i" % model_idx)
        model = auto_encoder_large(data[x_key][0].shape, compressed_size = options.AE_size)
        #model = auto_encoder(data[x_key][0].shape, compressed_size = options.AE_size)
        model.compile(optimizer=myoptimizer,loss= tf.keras.losses.mean_squared_error)
        if(model_idx == 0): model.summary()

        history = model.fit(t_data, 
                epochs = options.num_epoch, 
                validation_data = v_data,
                callbacks = cbs,
                verbose = 2 )


        histories.append(history)
        model_list.append(model)


    if(options.num_models == 1):
        best_model = model_list[0]
    else:
        min_loss = 9999999
        best_i = -1

        for model_idx in range(options.num_models):
            #preds = model_list[model_idx].predict(val_data[x_key])
            #loss = np.mean(np.mean(np.square(val_data[x_key] - preds), axis = (1,2)).reshape(-1))
            loss = histories[model_idx].history['val_loss'][-1]
            print("Model %i,  val loss %.4f " % (model_idx, loss))
            if(loss < min_loss):
                min_loss = loss
                best_i = model_idx
        print("Selecting model %i " % best_i)
        best_model = model_list[best_i]

    print("Saving model to : ", options.output)
    best_model.save(options.output)

    del data
    if(val_data is not None): del val_data

if(__name__ == "__main__"):
    parser = input_options()
    options = parser.parse_args()
    train_auto_encoder(options)
