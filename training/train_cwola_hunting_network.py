import sys
sys.path.append('..')
from utils.TrainingUtils import *
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Input, Flatten, Activation, Reshape, UpSampling2D
from tensorflow.keras.models import Model, Sequential, load_model
import h5py
from optparse import OptionParser
from optparse import OptionGroup



parser = OptionParser()
parser = OptionParser(usage="usage: %prog analyzer outputfile [options] \nrun with --help to get list of options")
parser.add_option("-i", "--fin", default='../data/jet_images.h5', help="Input file with data for training.")
parser.add_option("--plot_dir", default='../plots/', help="Directory to output plots")
parser.add_option("--model_dir", default='../models/', help="Directory to read in and output models")
parser.add_option("--model_name", default='cwbh.h5', help="What to name the model")
parser.add_option("--num_epoch", type = 'int', default=30, help="How many epochs to train for")

parser.add_option("--use_both", default = False, action = "store_true", help="Make a classifier for both jets together instead of just 1")
parser.add_option("-j", "--training_j", type ='int', default = 1, help="Which jet to make a classifier for (1 or 2)")
parser.add_option("--use_dense", default = False, action = "store_true", help="Make a classifier using inputs from cwola hunting paper instead of jet images")
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
batch_size = 200

sig_frac = -1.
signal = 1
if(options.filt_sig): sig_frac = options.sig_frac




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
    data = prepare_dataset(options.fin, signal_idx = signal, sig_frac = sig_frac )

    if(num_data < 0):
        num_data = data['label'].shape[0]

    if(use_both):
        j1s = data['j1_images'][data_start:data_start + num_data]
        j2s = data['j2_images'][data_start:data_start + num_data]
        X = np.stack((j1s,j2s), axis = -1)
        
    elif(options.training_j == 1):
        X = data['j1_images'][data_start:data_start + num_data]
        X = np.expand_dims(X, axis=-1)
    else:
        X = data['j2_images'][data_start:data_start + num_data]
        X = np.expand_dims(X, axis=-1)


    Y = data['label'][:num_data]

    if(sample_standardize):
        X = standardize(*zero_center(X))[0]

    Y = data['label'][data_start:data_start + num_data]
    mjj = data['mjj'][data_start:data_start + num_data]

    del data #save memory



Y_mjj_window = ((mjj > options.mjj_low) & (mjj < options.mjj_high)).reshape(-1)
window_size = (options.mjj_high - options.mjj_low)/ 2.
bkg_sample = (((mjj > (options.mjj_low - window_size)) & (mjj < options.mjj_low)) |
              ((mjj > options.mjj_high) & (mjj < (options.mjj_high + window_size))))
keep_event = bkg_sample | (Y_mjj_window == 1)

X = X[keep_event]
Y = Y[keep_event]
Y_mjj_window = Y_mjj_window[keep_event]







(X_train, X_val,
 Y_train, Y_val,
Y_train_true, Y_val_true) = sk.model_selection.train_test_split(X, Y_mjj_window, Y, test_size=val_frac)




print_signal_fractions(Y_train_true, Y_train)






myoptimizer = optimizers.Adam(lr=0.001, beta_1=0.8, beta_2=0.99, epsilon=1e-08, decay=0.0005)
if(options.use_dense):
    my_model = dense_net(X_train.shape[1])
else: 
    my_model = CNN(X_train[0].shape)
my_model.summary()
my_model.compile(optimizer=myoptimizer,loss='binary_crossentropy',
        metrics = ['accuracy'])
model_save_path = model_dir+ j_label+ model_name
print("Will save model to : ", model_save_path)


additional_val = AdditionalValidationSets([(X_val, Y_val_true, "Val_true_sig")], batch_size = 500)

cbs = [keras.callbacks.History()] 
roc = RocCallback(training_data=(X_train, Y_train), validation_data=(X_val, Y_val))

cbs.append(additional_val)
cbs.append(roc)




# train model
history = my_model.fit(X_train, Y_train,
          epochs=options.num_epoch,
          batch_size=batch_size,
          validation_data=(X_val, Y_val),
          callbacks = cbs,
          verbose=2)

plot_training(history.history, plot_dir + j_label + "training_history.png")
print("Saving model to : ", model_save_path)
my_model.save(model_save_path)
