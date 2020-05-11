import sys
sys.path.append('..')
from utils.TrainingUtils import *
import energyflow as ef
from energyflow.utils import data_split, pixelate, standardize, to_categorical, zero_center
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential, load_model
from sklearn.utils import shuffle as sk_shuffle
import h5py
from optparse import OptionParser
from optparse import OptionGroup



parser = OptionParser()
parser = OptionParser(usage="usage: %prog analyzer outputfile [options] \nrun with --help to get list of options")
parser.add_option("--fin", default='../data/jet_images_v3.h5', help="Input file for training.")
parser.add_option("--plot_dir", default='../plots/', help="Directory to output plots")
parser.add_option("--model_dir", default='../models/', help="Directory to read in and output models")
parser.add_option("--model_name", default='pure_cwola.h5', help="What to name the model")
parser.add_option("--num_epoch", type = 'int', default=20, help="How many epochs to train for")

parser.add_option("-j", "--training_j", type ='int', default = 1, help="Which jet to make a classifier for (1 or 2)")
parser.add_option("--use_dense", default = False, action = "store_true", help="Make a classifier using nsubjettiness inputs like cwbh instead of jet images")
parser.add_option("--old_dense", default = False, action = "store_true", help="Dense inputs are stored in older format (depricated)")

parser.add_option("-s", "--sig_frac", type = 'float', default = 0.06,  help="Signal fraction in signal-rich region")


(options, args) = parser.parse_args()


plot_dir = options.plot_dir
model_dir = options.model_dir




model_name = options.model_name

use_both = False



#################################################################


num_data = 200000
frac_sig_rich = 5
data_start = 0



val_frac = 0.1
test_frac = 0.0
batch_size = 200





#which images to train on and which to use for labelling
if(options.training_j == 1):
    j_label = "j1_"
    print("training classifier for j1")

elif (options.training_j ==2):
    j_label = "j2_"
    print("training classifier for j2")
else:
    print("Training jet not 1 or 2! Exiting")
    exit(1)


if(not options.use_dense):
    hf_in = h5py.File(options.fin, "r")
    if(use_both):
        j1s = hf_in['j1_images'][data_start:data_start + num_data]
        j2s = hf_in['j2_images'][data_start:data_start + num_data]
        #j1s = np.expand_dims(j1s, axis=-1)
        #j2s = np.expand_dims(j2s, axis=-1)
        X = np.stack((j1s,j2s), axis = -1)
        
    elif(options.training_j == 1):
        X = hf_in['j1_images'][data_start:data_start + num_data]
        X = np.expand_dims(X, axis=-1)
    else:
        X = hf_in['j2_images'][data_start:data_start + num_data]
        X = np.expand_dims(X, axis=-1)

    X = standardize(*zero_center(X))[0]
    jet_infos = hf_in['jet_infos'][data_start:data_start + num_data]
    Y = jet_infos[:,0] #is signal bit is first bit of info
    j1_4vec = jet_infos[:,1:5]
    j2_4vec = jet_infos[:,5:9]
    mjj = jet_infos[:,9]

else:
    if(not options.old_dense):
        hf_dense = h5py.File(options.fin, "r")
        dense_events = hf_dense['data']
        dense_vals = clean_events_v2(dense_events)
        idx1_start = 2
        idx1_end = 8
        idx2_start = 8
        idx2_end = 14
        dense_start = data_start
        print("Dense start is %i " % dense_start)
        if(dense_start < 0):
            print("Data start is %i, dense start event is %i, error, exiting \n" %data_start, dense_start_evt)
        mjj = dense_events[dense_start:dense_start + num_data, 1]
        Y = dense_events[dense_start:dense_start + num_data, 0]

        Y_mjj_window = ((mjj > options.mjj_low) & (mjj < options.mjj_high)).reshape(-1)
        if(options.use_both):
            X = dense_vals[dense_start:dense_start + num_data, idx1_start:idx2_end]
        elif(options.training_j == 1):
            X = dense_vals[dense_start:dense_start + num_data, idx1_start:idx1_end]
        elif(options.training_j ==2):
            X = dense_vals[dense_start:dense_start + num_data, idx1_end:idx2_end]


    else:
        pd_events = pd.read_hdf(options.fin)
        pd_events = clean_events(pd_events)
        idx1_start = 2
        idx1_end = 8
        idx2_start = 8
        idx2_end = 14
        if(options.use_both):
            X = pd_events.iloc[data_start:data_start + num_data, idx1_start:idx2_end].values
        elif(options.training_j == 1):
            X = pd_events.iloc[data_start:data_start + num_data, idx1_start:idx1_end].values
        elif(options.training_j ==2):
            X = pd_events.iloc[data_start:data_start + num_data, idx2_start:idx2_end].values

        Y = pd_events.iloc[data_start:data_start + num_data, [0]].values.reshape(-1)
        mjj = pd_events.iloc[data_start:data_start + num_data, [1]].values.reshape(-1)


num_sig_rich = num_data // frac_sig_rich

X_sig_rich = X[:num_sig_rich]
Y_sig_rich = Y[:num_sig_rich]
sig_mask = get_signal_mask(Y_sig_rich, options.sig_frac)
X_sig_rich = X_sig_rich[sig_mask]
Y_sig_rich = Y_sig_rich[sig_mask]

X_bkg_rich = X[num_sig_rich:]
Y_bkg_rich = Y[num_sig_rich:]
bkg_mask = get_signal_mask(Y_bkg_rich, 0.001)
X_bkg_rich = X_bkg_rich[bkg_mask]
Y_bkg_rich = Y_bkg_rich[bkg_mask]



X_cat = np.concatenate((X_sig_rich, X_bkg_rich))
Y_cat = np.concatenate((Y_sig_rich, Y_bkg_rich))
labels = np.concatenate((np.ones((X_sig_rich.shape[0]), dtype=np.float32), np.zeros((X_bkg_rich.shape[0]), dtype=np.float32)))

X_new, Y_new_true, Y_new  = sk_shuffle(X_cat, Y_cat, labels, random_state = 123)


print_signal_fractions(Y_new_true, Y_new)

(X_train, X_val, X_test, 
        Y_true_train, Y_true_val, Y_true_test,
        Y_train, Y_val, Y_test) = data_split(X_new, Y_new_true, Y_new, val=val_frac, test=test_frac, shuffle = False)






evt_weights = np.ones(X_train.shape[0])





myoptimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.8, beta_2=0.99, epsilon=1e-08, decay=0.0005)
if(options.use_dense):
    my_model = dense_net(X_train.shape[1])
else: 
    my_model = CNN(X_train[0].shape)
my_model.summary()
print("Input shape is ", my_model.layers[0].input_shape)
print("Data shape is ", X_train.shape)
my_model.compile(optimizer=myoptimizer,loss='binary_crossentropy',
          metrics = [keras.metrics.AUC()]
        )
model_save_path = model_dir+ j_label+ model_name
print("Will save model to : ", model_save_path)

checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_dir + j_label + "ckpt{epoch:02d}_"+model_name, 
        monitor='val_auc', verbose=1, save_best_only=False, save_weights_only=False, mode='max', period = 5)
early_stop = keras.callbacks.EarlyStopping(monitor='val_auc', min_delta=0, patience=10, verbose=1, mode='max', baseline=None, restore_best_weights=True)

additional_val = AdditionalValidationSets([(X_val, Y_true_val, "Val_true_sig")], batch_size = 500)

cbs = [keras.callbacks.History()] 
cbs.append(additional_val)




# train model
history = my_model.fit(X_train, Y_train,
          epochs=options.num_epoch,
          batch_size=batch_size,
          validation_data=(X_val, Y_val),
          sample_weight = evt_weights,
          callbacks = cbs,
          verbose=1)

plot_training(history.history, plot_dir + j_label + "training_history.png")
print("Saving model to : ", model_save_path)
my_model.save(model_save_path)
