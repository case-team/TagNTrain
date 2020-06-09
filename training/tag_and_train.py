import sys
sys.path.append('..')
from utils.TrainingUtils import *
from optparse import OptionParser
from optparse import OptionGroup

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
parser.add_option("--num_data", type='int', default=200000, help="How many events to use for training (before filtering)")
parser.add_option("--num_epoch", type = 'int', default=40, help="How many epochs to train for")

parser.add_option("-j", "--training_j", type ='int', default = 1, help="Which jet to make a classifier for (1 or 2)")
parser.add_option("--use_dense", default = False, action = "store_true", help="Make a classifier using nsubjettiness inputs like cwbh instead of jet images")
parser.add_option("--retrain", default = False, action = "store_true", help="Rather than initializing a new classifier, retrain an existing one")
parser.add_option("--reweight", default = False, action = "store_true", help="Reweight events in background region to match pT distribution of sig region")
parser.add_option("--start_cwbh", default = False, action = "store_true", help="In iteration 0, use cwola bumphunt network as initial labeler rather than autoencoder")
parser.add_option("--mjj_cut", default = False, action = "store_true", help="Require sig-like events to be in mass window")
parser.add_option("--mjj_cut_sideband", default = False, action = "store_true", help="Require bkg-like events to be in mass sidebands")
parser.add_option("--mjj_low", type='int', default = 3300,  help="Low mjj cut value")
parser.add_option("--mjj_high", type='int', default = 3700, help="High mjj cut value")
parser.add_option("--sig_cut", type='int', default = 80,  help="What classifier percentile to use to define sig-rich region")
parser.add_option("--bkg_cut", type='int', default = 40,  help="What classifier percentile to use to define bkg-rich region")

parser.add_option("-s", "--sig_frac", type = 'float', default = -1.,  help="Reduce signal to this amount (< 0 means don't filter)")
parser.add_option("--no_end_str", default = False, action = "store_true",  help="Don't do automatic end string based on signal frac")


(options, args) = parser.parse_args()


plot_dir = options.plot_dir
model_dir = options.model_dir


val_frac = 0.1
batch_size = 200
sample_standardize = False

signal = 1


#################################################################
end_str = ".h5"

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
    if(options.tnt_iter == 0):
        if(not options.start_cwbh):
            print("In interation 0. Using autoencoder as starting network")
            labeler_name = "auto_encoder" + end_str
        else :
            print("In interation 0. Using cwola bumphunt as starting network")
            labeler_name = "cwbh_dense" + end_str
    else:
        labeler_name = "TNT" + str(options.tnt_iter - 1) + "_" + network_type +  end_str
    f_labeler = model_dir + opp_j_label + labeler_name
else:
    f_labeler = options.labeler_name


if(options.model_name == ""):
    model_name = "TNT" + str(options.tnt_iter) + "_" + network_type + end_str
    f_model = model_dir+ j_label+ model_name
else:
    f_model = options.model_name

if(options.retrain):
    model_start = model_name
else:
    model_start = options.model_start

plot_prefix = "TNT" + str(options.tnt_iter) + "_" + network_type 

#start with different data not to overlap training sets
data_start = options.evt_offset + options.num_data * (options.tnt_iter + 1)
print("TNT iter is ", options.tnt_iter)
print("Will train using %i events, starting at event %i" % (options.num_data, data_start))




keep_low = keep_high = -1.

if(options.mjj_cut):
    window_size = (options.mjj_high - options.mjj_low)/2.
    keep_low = options.mjj_low - window_size
    keep_high = options.mjj_high + window_size
    print("Requiring mjj window from %.0f to %.0f \n" % (options.mjj_low, options.mjj_high))




import time
t1 = time.time()
data = DataReader(options.fin, signal_idx = signal, sig_frac = options.sig_frac, start = data_start, stop = data_start + options.num_data, m_low = keep_low, m_high = keep_high )
data.read()
#data = prepare_dataset(options.fin, signal_idx = signal, sig_frac = options.sig_frac,start = data_start, stop = data_start + options.num_data )
t2 = time.time()
print("load time  %s " % (t2 -t1))

X = data[j_label+'images']
L = data[opp_j_label+'images']

Y = data['label']
mjj = data['mjj']




mjj_window = ((mjj > options.mjj_low) & (mjj < options.mjj_high))


if(sample_standardize):
    X = standardize(*zero_center(X))[0]
    L = standardize(*zero_center(L))[0]


#save memory
del data    






labeler_plot = plot_dir+ opp_j_label +  plot_prefix + "_labeler_regions.png"
pt_plot = plot_dir + j_label + plot_prefix + "pt_dists.png"
pt_rw_plot = plot_dir + j_label + plot_prefix + "pt_rw_dists.png"
training_plot = plot_dir + j_label + plot_prefix + "training_history.png"


print("Loading labeling model from %s " % f_labeler)
labeler = load_model(f_labeler)

L_pred = labeler.predict(L, batch_size = 500)

print("Using model %s as labeler \n" % f_labeler)
if(len(L_pred.shape) > 2 ): #autoencoder
    L_labeler_scores = np.mean(np.square(L_pred - L), axis = (1,2)).reshape(-1)

else: 
    L_labeler_scores = L_pred.reshape(-1)


print("Sig-rich region defined > %i percentile" %options.sig_cut)
print("Bkg-rich region defined < %i percentile" %options.bkg_cut)

sig_region_cut = np.percentile(L_labeler_scores, options.sig_cut)
bkg_region_cut = np.percentile(L_labeler_scores, options.bkg_cut)

#make plot of labeler scores and where cuts are
#train_sig_events = (Y > 0.9).reshape(-1)
#train_bkg_events = (Y < 0.1).reshape(-1)
#
#scores = [L_labeler_scores[train_bkg_events], L_labeler_scores[train_sig_events]]
#labels = ['Background', 'Signal']
#colors = ['b', 'r']
#make_histogram(scores, labels, colors, 'Labeler Score', "", 100,
#               normalize = True, save = False)
#plt.axvline(linewidth=2, x=sig_region_cut, color ='black')
#plt.axvline(linewidth=2, x=bkg_region_cut, color = 'black')
#plt.savefig(labeler_plot)



(X, Y_true, mjj), Y_lab = sample_split(X, Y, mjj, cut_var = L_labeler_scores, sig_high = True, sig_cut = sig_region_cut, bkg_cut = bkg_region_cut)


print_signal_fractions(Y_true, Y_lab)
if(options.mjj_cut):
    outside_mjj_window = ((mjj < options.mjj_low) | (mjj > options.mjj_high))
    #TODO: Is this optimal? Should I filter these events out of the training set instead?
    Y_lab[outside_mjj_window] = 0
    print("After mass cut new sig fracs are:  ")
    print_signal_fractions(Y_true, Y_lab)

(X_train, X_val, 
        Y_true_train, Y_true_val,
        Y_lab_train, Y_lab_val) = train_test_split(X, Y_true, Y_lab, test_size=val_frac)

evt_weights = np.ones(X_train.shape[0])






myoptimizer = optimizers.Adam(lr=0.001, beta_1=0.8, beta_2=0.99, epsilon=1e-08, decay=0.0005)

if(model_start == ""):
    print("Creating new model ")
    if(options.use_dense): my_model = dense_net(X_train.shape[1])
    else: my_model = CNN(X_train[0].shape)

    my_model.summary()
    my_model.compile(optimizer=myoptimizer,loss='binary_crossentropy',
              metrics = ['accuracy']
            )
else:
    print("Starting with model from %s " % model_start)
    my_model = load_model(model_dir + j_label + model_start)



additional_val = AdditionalValidationSets([(X_val, Y_true_val, "Val_true_sig")], batch_size = 500)
roc1 = RocCallback(training_data=(X_train, Y_true_train), validation_data=(X_val, Y_true_val), extra_label = "true: ")
roc2 = RocCallback(training_data=(X_train, Y_lab_train), validation_data=(X_val, Y_lab_val), extra_label = "labeled: ")

cbs = [callbacks.History(), additional_val, roc1, roc2] 





# train model
history = my_model.fit(X_train, Y_lab_train,
          epochs=options.num_epoch,
          batch_size=batch_size,
          validation_data=(X_val, Y_lab_val),
          callbacks = cbs,
          verbose=2)

print("Saving model to : ", f_model)
my_model.save(f_model)
plot_training(history.history, fname = training_plot)
