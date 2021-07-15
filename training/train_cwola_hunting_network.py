import sys
sys.path.append('..')
from utils.TrainingUtils import *
from optparse import OptionParser
from optparse import OptionGroup
import random




parser = input_options()
(options, args) = parser.parse_args()


plot_dir = options.plot_dir




model_name = options.model_name







#################################################################





window_size = (options.mjj_high - options.mjj_low)/2.
window_frac = window_size / ((options.mjj_high + options.mjj_low)/ 2.)

#keep window size proportional to mjj bin center
window_low_size = window_frac*options.mjj_low / (1 + window_frac)
window_high_size = window_frac*options.mjj_high / (1 - window_frac)
options.keep_low = options.mjj_low - window_low_size
options.keep_high = options.mjj_high + window_high_size

print("Mjj keep low %.0f keep high %.0f \n" % ( options.keep_low, options.keep_high))


#which images to train on and which to use for labelling
if(not options.use_one):
    j_label = "jj_"
    img_key = 'jj_images'
    feat_key = 'jj_features'
    cnn_shape = (32,32,2)
    dense_shape = 16
    print("training classifier for both jets")
elif(options.training_j == 1):
    j_label = "j1_"
    img_key = 'j1_images'
    feat_key = 'j1_features'
    cnn_shape = (32,32,1)
    dense_shape = 8
    print("training classifier for j1")

elif (options.training_j ==2):
    j_label = "j2_"
    img_key = 'j2_images'
    feat_key = 'j2_features'
    cnn_shape = (32,32,1)
    dense_shape = 8
    print("training classifier for j2")
else:
    print("Training jet not 1 or 2! Exiting")
    exit(1)


options.keys = ['mjj']
if(options.use_one and not options.no_ptrw):
    options.keys.append('jet_kinematics')

if(not options.use_dense):
    import time
    options.keys.append(img_key)
    x_key = img_key

else:
    import time
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

print(mjjs[sig_events] [:10])

sample_weights = None
if(not options.no_sample_weights):
    sample_weights = "weight"
if(options.use_one and not options.no_ptrw): 
    sample_weights = j_label + 'ptrw'
    data.make_ptrw('Y_mjj', use_weights = not options.no_sample_weights, save_plots = False)
    #TODO, make validation use same reweighting ratios as training data?
    if(do_val): val_data.make_ptrw('Y_mjj', use_weights = not options.no_sample_weights, save_plots = False)



np.random.seed(options.seed)
tf.set_random_seed(options.seed)
os.environ['PYTHONHASHSEED']=str(options.seed)
random.seed(options.seed)


myoptimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.8, beta_2=0.99, epsilon=1e-08, decay=0.0005)

if(options.model_name == ""):
    f_model = options.model_dir+ j_label+ "cwbh.h5"
else:
    f_model = options.model_name
print("Will save model to : ", f_model)


#additional_val = AdditionalValidationSets([(X_val, Y_val_true, "Val_true_sig")], batch_size = 500)

cbs = [tf.keras.callbacks.History()]
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-6, patience=options.num_epoch/10, verbose=1, mode='min')
cbs.append(early_stop)





# train model
t_data = data.gen(x_key,'Y_mjj', key3 = sample_weights, batch_size = options.batch_size)
v_data = None
nVal = 0
if(do_val): 
    nVal = val_data.nTrain
    #v_data = val_data.gen(x_key,'label', key3 = sample_weights, batch_size = options.batch_size) #truth labels
    v_data = val_data.gen(x_key,'Y_mjj', key3 = sample_weights, batch_size = options.batch_size) #mjj labels

    roc = RocCallback(training_data=(np.zeros(100), np.zeros(100)), validation_data=(val_data[x_key], val_data['label']), extra_label = "true: ")
    cbs.append(roc)

print("Will train on %i events, validate on %i events" % (data.nTrain, nVal))
print(data[x_key][0].shape)

model_list = []

print("Will train %i models" % options.num_models)
for model_idx in range(options.num_models):
    print("Creating model %i" % model_idx)

    if(not options.use_dense):
        if(not options.large):
            model = CNN(cnn_shape)
        else:
            model = CNN_large(cnn_shape)
    else:
        model = dense_net(dense_shape)

    model.compile(optimizer=myoptimizer,loss='binary_crossentropy',
            metrics = ['accuracy'])
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

    val_sig_events = val_data['Y_mjj'] > 0.9
    val_bkg_events = val_data['Y_mjj'] < 0.1
    for model_idx in range(options.num_models):
        preds = model_list[model_idx].predict(val_data[x_key])
        loss = bce(preds.reshape(-1), val_data['Y_mjj'][()].reshape(-1), weights = val_data[sample_weights])
        true_loss = bce(preds.reshape(-1), val_data['label'][()].reshape(-1))
        auc = roc_auc_score(val_data['label'], preds)
        eff_cut_metric = compute_effcut_metric(preds[val_sig_events], preds[val_bkg_events], eff = 0.01)
        print("Model %i,  loss %.3f, true loss %.3f, auc %.3f, effcut metric %.3f" % (model_idx, loss, true_loss, auc, eff_cut_metric))
        #loss = -eff_cut_metric
        if(loss < min_loss):
            min_loss = loss
            best_i = model_idx
    print("Selecting model %i " % best_i)
    best_model = model_list[best_i]


if(do_val):
    msg = "End of training. "
    y_pred_val = best_model.predict_proba(val_data[x_key])
    roc_val = roc_auc_score(val_data['label'], y_pred_val)
    phrase = " roc-auc_val: %s (based on %i signal validation events)" % (str(round(roc_val,4)), np.sum(val_data['label']))
    msg += phrase
    print(msg, end =100*' ' + '\n')

#plot_training(history.history, options.plot_dir + j_label + "training_history.png")

print("Saving model to : ", f_model)
best_model.save(f_model)
data.cleanup()
