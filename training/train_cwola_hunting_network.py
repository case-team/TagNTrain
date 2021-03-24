import sys
sys.path.append('..')
from utils.TrainingUtils import *
from optparse import OptionParser
from optparse import OptionGroup



parser = OptionParser()
parser = OptionParser(usage="usage: %prog analyzer outputfile [options] \nrun with --help to get list of options")
parser.add_option("-i", "--fin", default='../data/jet_images.h5', help="Input file with data for training.")
parser.add_option("--plot_dir", default='../plots/', help="Directory to output plots")
parser.add_option("--model_dir", default='../models/', help="Directory to read in and output models")
parser.add_option("-o", "--model_name", default='cwbh.h5', help="What to name the model")
parser.add_option("--num_epoch", type = 'int', default=100, help="How many epochs to train for")
parser.add_option("--data_start", type = 'int', default=0, help="What event to start with")
parser.add_option("--num_data", type = 'int', default=-1, help="How many events to train on")
parser.add_option("--batch_start", type='int', default=-1, help="Train over multiple batches of dataset. Starting batch")
parser.add_option("--batch_size", type='int', default=256, help="Number size of mini-batchs used for training")
parser.add_option("--val_frac", type='float', default=0.1, help="Fraction of events used as validation")
parser.add_option("--batch_stop", type='int', default=-1, help="Train over multiple batches of dataset. Stopping batch (inclusive)")

parser.add_option("--use_one", default = False, action = "store_true", help="Make a classifier for one jet instead of both")
parser.add_option("-j", "--training_j", type ='int', default = 1, help="Which jet to make a classifier for (1 or 2)")
parser.add_option("--use_dense", default = False, action = "store_true", help="Make a classifier using inputs from cwola hunting paper instead of jet images")
parser.add_option("--mjj_low", type='int', default = 2250,  help="Low mjj cut value")
parser.add_option("--mjj_high", type='int', default = 2750, help="High mjj cut value")
parser.add_option("--mjj_sig", type='int', default = 2500, help="Signal mass (used for signal filtering)")
parser.add_option("--d_eta", type='float', default = -1, help="Delta eta cut")
parser.add_option("--no_ptrw", default = False, action="store_true",  help="Don't reweight events to have matching pt distributions in sig-rich and bkg-rich samples")

parser.add_option("--large", default = False, action = "store_true", help="Use larger NN archetecture")
parser.add_option("--sig_idx", type = 'int', default = 1,  help="What index of signal to use")
parser.add_option("-s", "--sig_frac", type = 'float', default = -1.,  help="Reduce signal to this amount (< 0 to not filter )")
parser.add_option("--hadronic_only",  default=False, action='store_true',  help="Filter out leptonic decays of signal")
parser.add_option("--seed", type = 'int', default = 123456,  help="RNG seed for model")


(options, args) = parser.parse_args()


plot_dir = options.plot_dir
model_dir = options.model_dir




model_name = options.model_name






#################################################################





val_frac = options.val_frac
batch_size = options.batch_size



window_size = (options.mjj_high - options.mjj_low)/2.
keep_low = options.mjj_low - window_size
keep_high = options.mjj_high + window_size


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


keys = ['mjj']
if(options.use_one and not options.no_ptrw):
    keys.append('jet_kinematics')

if(not options.use_dense):
    import time
    keys.append(img_key)

else:
    import time
    keys.append(feat_key)
    
t1 = time.time()
data = DataReader(options.fin, keys = keys, signal_idx = options.sig_idx, sig_frac = options.sig_frac, start = options.data_start, stop = options.data_start + options.num_data, 
        m_low = keep_low, m_high = keep_high, val_frac = val_frac, batch_start = options.batch_start, batch_stop = options.batch_stop , hadronic_only = options.hadronic_only, 
        m_sig = options.mjj_sig, seed = options.seed, eta_cut = options.d_eta)
data.read()
data.make_Y_mjj(options.mjj_low, options.mjj_high)
t2 = time.time()
print("load time  %s " % (t2 -t1))




print_signal_fractions(data['label'], data['Y_mjj'])

sample_weights = None
if(options.use_one and not options.no_ptrw): 
    sample_weights = j_label + 'ptrw'
    data.make_ptrw('Y_mjj', save_plots = False)



np.random.seed(options.seed)


myoptimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.8, beta_2=0.99, epsilon=1e-08, decay=0.0005)
if(not options.use_dense):
    x_key = img_key
    if(not options.large):
        my_model = CNN(cnn_shape)
    else:
        my_model = CNN_large(cnn_shape)
else:
    x_key = feat_key
    my_model = dense_net(dense_shape)

my_model.compile(optimizer=myoptimizer,loss='binary_crossentropy',
        metrics = ['accuracy'])
my_model.summary()

if(options.model_name == ""):
    f_model = model_dir+ j_label+ "cwbh.h5"
else:
    f_model = options.model_name
print("Will save model to : ", f_model)


#additional_val = AdditionalValidationSets([(X_val, Y_val_true, "Val_true_sig")], batch_size = 500)

cbs = [tf.keras.callbacks.History()]
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-6, patience=10, verbose=1, mode='min')
cbs.append(early_stop)





# train model
t_data = data.gen(x_key,'Y_mjj', key3 = sample_weights, batch_size = batch_size)
v_data = None
if(val_frac > 0.): 
    v_data = data.gen('val_'+x_key,'val_label', batch_size = batch_size)
    roc = RocCallback(training_data=(np.zeros(100), np.zeros(100)), validation_data=(data['val_'+x_key], data['val_label']), extra_label = "true: ")
    cbs.append(roc)

print("Will train on %i events, validate on %i events" % (data.nTrain, data.nVal))
#print(np.mean(data['val_label']))
#print(np.mean(data['label']))

print(data[x_key][0].shape)

history = my_model.fit(t_data, 
        epochs = options.num_epoch, 
        validation_data = v_data,
        callbacks = cbs,
        verbose = 2 )

if(np.sum(data['val_label']) > 10):
    msg = "End of training. "
    y_pred_val = my_model.predict_proba(data['val_'+x_key])
    roc_val = roc_auc_score(data['val_label'], y_pred_val)
    phrase = " roc-auc_val: %s (based on %i signal validation events)" % (str(round(roc_val,4)), np.sum(data['val_label']))
    msg += phrase
    print(msg, end =100*' ' + '\n')

plot_training(history.history, plot_dir + j_label + "training_history.png")
print("Saving model to : ", f_model)
my_model.save(f_model)
data.cleanup()
