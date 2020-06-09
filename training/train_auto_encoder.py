import sys
sys.path.append('..')
from utils.TrainingUtils import *
from optparse import OptionParser
from optparse import OptionGroup




parser = OptionParser()
parser = OptionParser(usage="usage: %prog analyzer outputfile [options] \nrun with --help to get list of options")
parser.add_option("-i", "--fin", default='../data/jet_images.h5', help="Input file for training.")
parser.add_option("--plot_dir", default='../plots/', help="Directory to output plots")
parser.add_option("-o", "--model_name", default='auto_encoder.h5', help="What to name the model")
parser.add_option("--num_epoch", type = 'int', default=30, help="How many epochs to train for")
parser.add_option("--num_data", type='int', default=200000, help="How many events to use for training (before filtering)")
parser.add_option("--model_start", default="", help="Starting point for model (empty string for new model)")

parser.add_option("-j", "--training_j", type ='int', default = 1, help="Which jet to make a classifier for (1 or 2)")

parser.add_option("--filt_sig", default = False, action = "store_true", help="Reduce the amount of signal in the dataset")
parser.add_option("-s", "--sig_frac", type = 'float', default = 0.01,  help="Reduce signal to this amount (default is 0.01)")

(options, args) = parser.parse_args()


draw_images = False
sample_standardize = False



val_frac = 0.1
batch_size = 200

use_j1 = (options.training_j == 1)


sig_frac = -1.
signal = 1
if(options.filt_sig): sig_frac = options.sig_frac

if(use_j1):
    j_label = "j1_"
else:
    j_label = "j2_"

print("Training autoencoder on %s !" % j_label[:-1])

img_key = j_label + "images"

data = prepare_dataset(options.fin, signal_idx = signal, keys = [img_key], sig_frac = sig_frac, start = 0,stop = options.num_data )


Y = data['label']



(X_train, X_val, Y_train, Y_val) = train_test_split(data[img_key], Y, test_size = val_frac)

if(sample_standardize):
    X_train, X_val, X_test = standardize(*zero_center(X_train, X_val, X_test))


print("Signal frac is %.3f \n" % (np.mean(Y_train)))

if(options.model_start == ""):
    print("Creating new model ")
    model = auto_encoder(X_train[0].shape)
    model.summary()
    myoptimizer = optimizers.Adam(lr=0.001, beta_1=0.8, beta_2=0.99, epsilon=1e-08, decay=0.0005)
    model.compile(optimizer=myoptimizer,loss=losses.mean_squared_error)
else:
    print("Starting with model from %s " % model_start)
    model = load_model(options.model_start)


early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='min', baseline=None, restore_best_weights=True)
cbs = [callbacks.History(), early_stop]

# train model
history = model.fit(X_train, X_train,
          epochs=options.num_epoch,
          batch_size=batch_size,
          validation_data=(X_val, X_val),
          callbacks = cbs,
          verbose=2)


print("Saving model to : ", options.model_name)
model.save(options.model_name)
# get predictions on test data

#X_test = X_val
#Y_test = Y_val
#X_reco_test = model.predict(X_test, batch_size=1000)
#X_reco_loss_test = np.mean(np.square(X_test - X_reco_test), axis = (1,2))
#
#test_sig_events = (Y_test == 1.)
#test_bkg_events = (Y_test == 0.)
#
#scores = [X_reco_loss_test[test_bkg_events], X_reco_loss_test[test_sig_events]]
#labels = ['background', 'signal']
#colors = ['b', 'r']
#save_figs = True
#make_histogram(scores, labels, colors, 'AutoEncoder Loss', "", 20,
#               normalize = True, save = save_figs, fname=options.plot_dir+j_label+"reco_loss.png")
#
#make_roc_curve([X_reco_loss_test], Y_test,  save = True, fname=options.plot_dir+j_label+"auto_encoder_roc.png")
#
