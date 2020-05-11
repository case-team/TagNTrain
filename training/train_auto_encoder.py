import sys
sys.path.append('..')
from utils.TrainingUtils import *
import energyflow as ef
from energyflow.utils import data_split, pixelate, standardize, to_categorical, zero_center
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential, load_model
import h5py
from optparse import OptionParser
from optparse import OptionGroup




parser = OptionParser()
parser = OptionParser(usage="usage: %prog analyzer outputfile [options] \nrun with --help to get list of options")
parser.add_option("--fin", default='../data/jet_images.h5', help="Input file for training.")
parser.add_option("--plot_dir", default='../plots/', help="Directory to output plots")
parser.add_option("--model_dir", default='../models/', help="Directory to read in and output models")
parser.add_option("--model_name", default='auto_encoder.h5', help="What to name the model")
parser.add_option("--num_epoch", type = 'int', default=30, help="How many epochs to train for")

parser.add_option("-j", "--training_j", type ='int', default = 1, help="Which jet to make a classifier for (1 or 2)")

parser.add_option("--filt_sig", default = False, action = "store_true", help="Reduce the amount of signal in the dataset")
parser.add_option("-s", "--sig_frac", type = 'float', default = 0.01,  help="Reduce signal to this amount (default is 0.01)")

(options, args) = parser.parse_args()
plot_dir = options.plot_dir
model_dir = options.model_dir
model_start = ""


draw_images = False

num_data = 200000


npix = 40
input_shape = (npix, npix)
val_frac = 0.1
test_frac = 0.0
batch_size = 200

use_j1 = (options.training_j == 1)




hf_in = h5py.File(options.fin, "r")

j1_images = hf_in['j1_images'][:num_data]
j1_images = np.expand_dims(j1_images, axis=-1)
j2_images = hf_in['j2_images'][:num_data]
j2_images = np.expand_dims(j2_images, axis=-1)
jet_infos = hf_in['jet_infos'][:num_data]
Y = jet_infos[:,0] #is signal bit is first bit of info

if(use_j1):
    images = j1_images
    j_label = "j1_"
    print("Training auto encoder on leading jet! label = j1")
else:
    images = j1_images
    j_label = "j2_"
    print("Training auto encoder on sub-leading jet! label = j2")



if(draw_images):
    images = np.squeeze(np.array(images))
    signal_bits = np.array(Y)
    signal_mask = (signal_bits == 1.)
    bkg_mask = (signal_bits == 0.)
    signal_images = images[signal_mask]
    mean_signal1 = np.mean(signal_images, axis=0)

    bkg_images = images[bkg_mask]
    mean_bkg1 = np.mean(bkg_images, axis=0)

    print("Avg pixel Sums: ", np.sum(mean_signal1), np.sum(mean_bkg1) )
    fig, (ax1, ax2) = plt.subplots(1,2)

    ax1.imshow(mean_signal1, cmap='gray') #, vmin=0., vmax=max_pix)
    ax1.set_title("Signal: Leading Jet")
    ax2.imshow(mean_bkg1, cmap='gray')#, vmin =0., vmax=max_pix)
    ax2.set_title("Background: Leading Jet")
    plt.show()


(X_train, X_val, X_test, Y_train, Y_val, Y_test) = data_split(images, Y, val=val_frac, test=test_frac, shuffle = True)

X_train, X_val, X_test = standardize(*zero_center(X_train, X_val, X_test))


if(options.filt_sig):
    train_mask = get_signal_mask(Y_train, options.sig_frac)
    val_mask = get_signal_mask(Y_val, options.sig_frac)

    X_train = X_train[train_mask]
    Y_train = Y_train[train_mask]
    X_val = X_val[val_mask]
    Y_val = Y_val[val_mask]


print("Signal frac is %.3f \n" % (np.mean(Y_train)))

if(model_start == ""):
    print("Creating new model ")
    model = auto_encoder(X_train[0].shape)
    model.summary()
    myoptimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.8, beta_2=0.99, epsilon=1e-08, decay=0.0005)
    model.compile(optimizer=myoptimizer,loss=keras.losses.mean_squared_error)
else:
    print("Starting with model from %s " % model_start)
    model = load_model(model_dir + j_label + model_start)


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='min', baseline=None, restore_best_weights=True)
cbs = [keras.callbacks.History(), early_stop]

# train model
history = model.fit(X_train, X_train,
          epochs=options.num_epoch,
          batch_size=batch_size,
          validation_data=(X_val, X_val),
          callbacks = cbs,
          verbose=1)

model.save(model_dir+j_label+options.model_name)
# get predictions on test data
X_test = X_val
Y_test = Y_val
X_reco_test = model.predict(X_test, batch_size=1000)
X_reco_loss_test =  np.mean(keras.losses.mean_squared_error(X_reco_test, X_test), axis=(1,2))

test_sig_events = (Y_test == 1.)
test_bkg_events = (Y_test == 0.)
print(X_reco_loss_test[test_bkg_events].shape, X_reco_loss_test[test_sig_events].shape)

scores = [X_reco_loss_test[test_bkg_events], X_reco_loss_test[test_sig_events]]
labels = ['background', 'signal']
colors = ['b', 'r']
save_figs = True
make_histogram(scores, labels, colors, 'AutoEncoder Loss', "", 20,
               normalize = True, save = save_figs, fname=plot_dir+j_label+"reco_loss.png")

make_roc_curve([X_reco_loss_test], Y_test,  save = True, fname=plot_dir+j_label+"auto_encoder_roc.png")

