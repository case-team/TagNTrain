import sys
sys.path.append('..')
from utils.TrainingUtils import *
import energyflow as ef
from energyflow.utils import data_split, pixelate, standardize, to_categorical, zero_center
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Input, Flatten, Activation, Reshape, UpSampling2D
from tensorflow.keras.models import Model, Sequential, load_model
import h5py
from optparse import OptionParser
from optparse import OptionGroup



parser = OptionParser()
parser = OptionParser(usage="usage: %prog analyzer outputfile [options] \nrun with --help to get list of options")
parser.add_option("--fin", default='../data/jet_images.h5', help="Input file with data for training.")
parser.add_option("--plot_dir", default='../plots/', help="Directory to output plots")
parser.add_option("--model_dir", default='../models/', help="Directory to read in and output models")
parser.add_option("--model_name", default='test.h5', help="What to name the model")
parser.add_option("--num_epoch", type = 'int', default=30, help="How many epochs to train for")

parser.add_option("--use_both", default = False, action = "store_true", help="Make a classifier for both jets together instead of just 1")
parser.add_option("-j", "--training_j", type ='int', default = 1, help="Which jet to make a classifier for (1 or 2)")
parser.add_option("--use_dense", default = False, action = "store_true", help="Make a classifier using inputs from cwola hunting paper instead of jet images")
parser.add_option("--old_dense", default = False, action = "store_true", help="Dense inputs are stored in older format (depricated)")
parser.add_option("--mjj_low", type='int', default = 3300,  help="Low mjj cut value")
parser.add_option("--mjj_high", type='int', default = 3700, help="High mjj cut value")

parser.add_option("--filt_sig", default = False, action = "store_true", help="Reduce the amount of signal in the dataset")
parser.add_option("-s", "--sig_frac", type = 'float', default = 0.01,  help="Reduce signal to this amount (default is 0.01)")


(options, args) = parser.parse_args()


plot_dir = options.plot_dir
model_dir = options.model_dir



use_both = True

model_name = options.model_name





#################################################################


num_data = 200000
data_start = 0



val_frac = 0.1
test_frac = 0.0
batch_size = 100





#which images to train on and which to use for labelling
if(options.use_both):
    j_label = "jj_"
    print("training classifier for both jets")
elif(options.training_j == 1):
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


if(options.filt_sig):
    print("Filtering sig to be %.3f" %options.sig_frac)
    mask = get_signal_mask(Y, options.sig_frac)

    X = X[mask]
    Y = Y[mask]
    mjj = mjj[mask]


Y_mjj_window = ((mjj > options.mjj_low) & (mjj < options.mjj_high)).reshape(-1)
window_size = (options.mjj_high - options.mjj_low)/ 2.
bkg_sample = (((mjj > (options.mjj_low - window_size)) & (mjj < options.mjj_low)) |
              ((mjj > options.mjj_high) & (mjj < (options.mjj_high + window_size))))
keep_event = bkg_sample | (Y_mjj_window == 1)

#print(window_size)
#print(mjj[:20])
#print(keep_event[:20])
#print(X.shape, Y_mjj_window.shape)
X = X[keep_event]
Y = Y[keep_event]
Y_mjj_window = Y_mjj_window[keep_event]
#print(X.shape, Y_mjj_window.shape)







(X_train, X_val, X_test, 
        Y_train, Y_val, Y_test, 
Y_train_true, Y_val_true, Y_test_true) = data_split(X, Y_mjj_window, Y, val=val_frac, test=test_frac, shuffle = True)




print_signal_fractions(Y_train_true, Y_train)

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

additional_val = AdditionalValidationSets([(X_val, Y_val_true, "Val_true_sig")], batch_size = 500)

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
