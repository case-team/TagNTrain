import sys
sys.path.append('..')
from utils.TrainingUtils import *
from optparse import OptionParser
from optparse import OptionGroup
import random

import h5py

# Options
parser = OptionParser()
parser = OptionParser(usage="usage: %prog analyzer outputfile [options] \nrun with --help to get list of options")
parser.add_option("--plot_dir", default='../plots/', help="Directory to output plots")
parser.add_option("--model_dir", default='../models/', help="Directory to read in and output models")
parser.add_option("-i", "--fin", default='../data/jet_images.h5', help="Input file for training.")
parser.add_option("-l", "--labeler_name", default='', help="What model to use to as the initial classifer")
parser.add_option("-o", "--model_name", default='', help="What to call the model (default based on tnt iteration)")
parser.add_option("--model_start", default="", help="Starting point for model (empty string for new model)")

parser.add_option("--iter", dest = "tnt_iter", type = 'int', default=0, 
        help="What iteration of  the tag & train algorithm this is (Start = 0).")
parser.add_option("--evt_offset", type='int', default=0, help="Offset to set which events to use for training")
parser.add_option("--num_data", type='int', default=-1, help="How many events to use for training (before filtering)")
parser.add_option("--data_start", type = 'int', default=0, help="What event to start with (-1 for trying to do it automatically based on iter)")
parser.add_option("--batch_start", type='int', default=-1, help="Train over multiple batches of dataset. Starting batch")
parser.add_option("--batch_stop", type='int', default=-1, help="Train over multiple batches of dataset. Stopping batch (inclusive)")
parser.add_option("--num_epoch", type = 'int', default=100, help="How many epochs to train for")
parser.add_option("--batch_size", type='int', default=256, help="Number size of mini-batchs used for training")
parser.add_option("--val_frac", type='float', default=0.1, help="Fraction of events used as validation")

parser.add_option("-j", "--training_j", type ='int', default = 1, help="Which jet to make a classifier for (1 or 2)")
parser.add_option("--use_dense", default = False, action = "store_true", help="Make a classifier using nsubjettiness inputs like cwbh instead of jet images")
parser.add_option("--retrain", default = False, action = "store_true", help="Rather than initializing a new classifier, retrain an existing one")
parser.add_option("--reweight", default = False, action = "store_true", help="Reweight events in background region to match pT distribution of sig region")
parser.add_option("--start_cwbh", default = False, action = "store_true", help="In iteration 0, use cwola bumphunt network as initial labeler rather than autoencoder")
parser.add_option("--no_mjj_cut", default = False, action = "store_true", help="Don't require a mass window")
parser.add_option("--mjj_low", type='int', default = 2250,  help="Low mjj cut value")
parser.add_option("--mjj_high", type='int', default = 2750, help="High mjj cut value")
parser.add_option("--mjj_sig", type='int', default = 2500, help="Signal mass (used in signal filtering)")
parser.add_option("--d_eta", type='float', default = -1, help="Delta eta cut")
parser.add_option("--sig_cut", type='int', default = 80,  help="What classifier percentile to use to define sig-rich region")
parser.add_option("--bkg_cut", type='int', default = 40,  help="What classifier percentile to use to define bkg-rich region")
parser.add_option("--no_ptrw", default = False, action="store_true",  help="Don't reweight events to have matching pt distributions in sig-rich and bkg-rich samples")

parser.add_option("-s", "--sig_frac", type = 'float', default = -1.,  help="Reduce signal to this amount (< 0 means don't filter)")
parser.add_option("--seed", type = 'int', default = 123456,  help="RNG seed for model")
parser.add_option("--sig_idx", type = 'int', default = 1,  help="What index of signal to use")
parser.add_option("--hadronic_only",  default=False, action='store_true',  help="Filter out leptonic decays of signal")


(options, args) = parser.parse_args()


plot_dir = options.plot_dir
model_dir = options.model_dir


val_frac = options.val_frac
batch_size = options.batch_size
cnn_shape = (32,32,1)
dense_shape = 8


#################################################################

np.random.seed(options.seed)
tf.set_random_seed(options.seed)
os.environ['PYTHONHASHSEED']=str(options.seed)
random.seed(options.seed)



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
    f_model = model_dir+ j_label+ model_name
else:
    f_model = options.model_name

if(options.retrain):
    model_start = model_name
else:
    model_start = options.model_start

plot_prefix = "TNT" + str(options.tnt_iter) + "_" + network_type 

#start with different data not to overlap training sets
data_start = options.data_start
if(data_start < 0):
    data_start = options.evt_offset + options.num_data * (options.tnt_iter + 1)
print("TNT iter is ", options.tnt_iter)




keep_low = keep_high = -1.

if(options.num_data == -1):
    data_stop = -1
else:
    data_stop = data_start + options.num_data
    print("Will train using %i events, starting at event %i" % (options.num_data, data_start))

if(not options.no_mjj_cut):
    window_size = (options.mjj_high - options.mjj_low)/2.
    keep_low = options.mjj_low - window_size
    keep_high = options.mjj_high + window_size
    print("Requiring mjj window from %.0f to %.0f \n" % (options.mjj_low, options.mjj_high))



keys = []
if(not options.use_dense):
    import time
    keys = ['mjj', 'j1_images', 'j2_images']
    if(not options.no_ptrw): keys.append('jet_kinematics')
    x_key = j_label +  'images'
    l_key = opp_j_label +  'images'
else:
    import time
    keys = ['mjj', 'j1_features', 'j2_features']
    if(not options.no_ptrw): keys.append('jet_kinematics')
    x_key = j_label +  'features'
    if('auto' in options.labeler_name):
        keys.append(opp_j_label + "images")
        l_key = opp_j_label + 'images'
    else:
        l_key = opp_j_label +  'features'

t1 = time.time()
data = DataReader(options.fin, keys = keys, signal_idx = options.sig_idx, sig_frac = options.sig_frac, start = data_start, stop = data_stop, 
        m_low = keep_low, m_high = keep_high, val_frac = val_frac, batch_start = options.batch_start, batch_stop = options.batch_stop, hadronic_only = options.hadronic_only,
        m_sig = options.mjj_sig, seed = options.seed, eta_cut = options.d_eta)
data.read()
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


print_signal_fractions(data['label'], data['Y_TNT'])


sample_weights = None
if(not options.no_ptrw): 
    sample_weights = j_label + 'ptrw'

    data.make_ptrw('Y_TNT', save_plots = False)





myoptimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.8, beta_2=0.99, epsilon=1e-08, decay=0.0005)


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
    my_model = tf.keras.models.load_model(model_dir + j_label + model_start)



#additional_val = AdditionalValidationSets([(X_val, Y_true_val, "Val_true_sig")], batch_size = 500)

#cbs = [callbacks.History(), additional_val, roc1, roc2] 


cbs = [tf.keras.callbacks.History()]
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-6, patience=10, verbose=1, mode='min')
cbs.append(early_stop)



# train model
t_data = data.gen(x_key,'Y_TNT', key3 = sample_weights,  batch_size = batch_size)
v_data = None
if(val_frac > 0.): 
    v_data = data.gen('val_'+x_key,'val_label', batch_size = batch_size)
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
training_plot = plot_dir + j_label + plot_prefix + "training_history.png"
plot_training(history.history, fname = training_plot)
data.cleanup()
