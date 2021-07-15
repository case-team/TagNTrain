import sys
sys.path.append('..')
from utils.TrainingUtils import *
from optparse import OptionParser
from optparse import OptionGroup
import random

import h5py

# Options
parser = input_options()
(options, args) = parser.parse_args()



cnn_shape = (32,32,1)
dense_shape = 8


#################################################################




if(options.use_dense): network_type = "dense"
else: network_type = "CNN"

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


if(options.model_name == ""):
    model_name = "TNT" + str(options.tnt_iter) + "_" + network_type + ".h5"
    f_model = options.model_dir+ j_label+ model_name
else:
    f_model = options.model_name

model_start = options.model_start

plot_prefix = "TNT" + str(options.tnt_iter) + "_" + network_type 

#start with different data not to overlap training sets
data_start = options.data_start
print("TNT iter is ", options.tnt_iter)




options.keep_low = options.keep_high = -1.

if(not options.no_mjj_cut):
    #keep window size proportional to mjj bin center
    window_low_size = window_frac*options.mjj_low / (1 + window_frac)
    window_high_size = window_frac*options.mjj_high / (1 - window_frac)
    options.keep_low = options.mjj_low - window_low_size
    options.keep_high = options.mjj_high + window_high_size

    print("Mjj keep low %.0f keep high %.0f \n" % ( options.keep_low, options.keep_high))



options.keys = []
if(not options.use_dense):
    options.keys = ['mjj', 'j1_images', 'j2_images']
    if(not options.no_ptrw): keys.append('jet_kinematics')
    x_key = j_label +  'images'
    l_key = opp_j_label +  'images'
else:
    options.keys = ['mjj', 'j1_features', 'j2_features']
    if(not options.no_ptrw): keys.append('jet_kinematics')
    x_key = j_label +  'features'
    if('auto' in options.labeler_name):
        options.keys.append(opp_j_label + "images")
        l_key = opp_j_label + 'images'
    else:
        l_key = opp_j_label +  'features'

import time
t1 = time.time()

#load the dataset
t1 = time.time()
data, val_data = load_dataset_from_options(options)
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

print("cut high %.3e, cut low %.3e " % (sig_region_cut, bkg_region_cut))

data.make_Y_TNT(sig_region_cut = sig_region_cut, bkg_region_cut = bkg_region_cut, cut_var = labeler_scores, mjj_low = options.mjj_low, mjj_high = options.mjj_high)

if(do_val):
    val_labeler_scores = val_data.labeler_scores(labeler,  l_key)
    val_data.make_Y_TNT(sig_region_cut = sig_region_cut, bkg_region_cut = bkg_region_cut, cut_var = val_labeler_scores, mjj_low = options.mjj_low, mjj_high = options.mjj_high)


print_signal_fractions(data['label'], data['Y_TNT'])


sample_weights = None
if(not options.no_ptrw): 
    sample_weights = j_label + 'ptrw'

    data.make_ptrw('Y_TNT', save_plots = False)
    if(do_val): val_data.make_ptrw('Y_TNT', save_plots = False)





myoptimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.8, beta_2=0.99, epsilon=1e-08, decay=0.0005)

np.random.seed(options.seed)
tf.set_random_seed(options.seed)
os.environ['PYTHONHASHSEED']=str(options.seed)
random.seed(options.seed)

if(model_start == ""):
    print("Creating new model based on seed %i " % options.seed)
    if(not options.use_dense):
        my_model = CNN(cnn_shape)
    else:
        my_model = dense_net(dense_shape)

    my_model.summary()
    my_model.compile(optimizer=myoptimizer,loss='binary_crossentropy',
              metrics = ['accuracy']
            )
else:
    print("Starting with model from %s " % model_start)
    my_model = tf.keras.models.load_model(options.model_dir + j_label + model_start)



#additional_val = AdditionalValidationSets([(X_val, Y_true_val, "Val_true_sig")], options.batch_size = 500)

#cbs = [callbacks.History(), additional_val, roc1, roc2] 


cbs = [tf.keras.callbacks.History()]
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-6, patience=10, verbose=1, mode='min')
cbs.append(early_stop)



# train model
t_data = data.gen(x_key,'Y_TNT', key3 = sample_weights,  batch_size = options.batch_size)
v_data = None
if(do_val): 
    v_data = data.gen('val_'+x_key,'val_label', batch_size = options.batch_size)
    roc = RocCallback(training_data=(np.zeros(100), np.zeros(100)), validation_data=(data['val_'+x_key], data['val_label']), extra_label = "true: ")
    cbs.append(roc)

print("Will train on %i events, validate on %i events" % (data.nTrain, data.nVal))



history = my_model.fit(t_data, 
        epochs = options.num_epoch, 
        validation_data = v_data,
        callbacks = cbs,
        verbose = 2 )

preds = my_model.predict_proba(data[x_key][:10])
if np.any(np.isnan(preds)): 
    print("Got output Nans. Should rerun with a different seed")
    sys.exit(1)


if(np.sum(data['val_label']) > 10):
    msg = "End of training. "
    y_pred_val = my_model.predict_proba(data['val_'+x_key])
    roc_val = roc_auc_score(data['val_label'], y_pred_val)
    phrase = " roc-auc_val: %s (based on %i signal validation events)" % (str(round(roc_val,4)), np.sum(data['val_label']))
    msg += phrase
    print(msg, end =100*' ' + '\n')
print("Saving model to : ", f_model)
my_model.save(f_model)
training_plot = options.plot_dir + j_label + plot_prefix + "training_history.png"
plot_training(history.history, fname = training_plot)
data.cleanup()
