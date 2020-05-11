import sys
sys.path.append('..')
from utils.TrainingUtils import *
import energyflow as ef
from energyflow.utils import data_split,  standardize, zero_center
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential, load_model
from optparse import OptionParser
from optparse import OptionGroup

import h5py

# Options
parser = OptionParser()
parser = OptionParser(usage="usage: %prog analyzer outputfile [options] \nrun with --help to get list of options")
parser.add_option("--plot_dir", default='../plots/', help="Directory to output plots")
parser.add_option("--model_dir", default='../models/', help="Directory to read in and output models")
parser.add_option("--fin", default='../data/jet_images.h5', help="Input file for training.")
parser.add_option("--extra_label", default='', help="Extra string to add to name of models")
parser.add_option("--model_start", default="", help="Starting point for model (empty string for new model)")

parser.add_option("-i", "--iter", dest = "tnt_iter", type = 'int', default=0, 
        help="What iteration of  the tag & train algorithm this is (Start = 0).")
parser.add_option("--evt_offset", type='int', default=0, help="Offset to set which events to use for training")
parser.add_option("--num_data", type='int', default=200000, help="How many events to use for training (before filtering)")
parser.add_option("--num_epoch", type = 'int', default=40, help="How many epochs to train for")

parser.add_option("-j", "--training_j", type ='int', default = 1, help="Which jet to make a classifier for (1 or 2)")
parser.add_option("--use_dense", default = False, action = "store_true", help="Make a classifier using nsubjettiness inputs like cwbh instead of jet images")
parser.add_option("--old_dense", default = False, action = "store_true", help="Dense inputs are stored in older format (depricated)")
parser.add_option("--retrain", default = False, action = "store_true", help="Rather than initializing a new classifier, retrain an existing one")
parser.add_option("--reweight", default = False, action = "store_true", help="Reweight events in background region to match pT distribution of sig region")
parser.add_option("--start_cwbh", default = False, action = "store_true", help="In iteration 0, use cwola bumphunt network as initial labeler rather than autoencoder")
parser.add_option("--mjj_cut", default = False, action = "store_true", help="Require sig-like events to be in mass window")
parser.add_option("--mjj_cut_sideband", default = False, action = "store_true", help="Require bkg-like events to be in mass sidebands")
parser.add_option("--mjj_low", type='int', default = 3300,  help="Low mjj cut value")
parser.add_option("--mjj_high", type='int', default = 3700, help="High mjj cut value")
parser.add_option("--sig_cut", type='int', default = 80,  help="What classifier percentile to use to define sig-rich region")
parser.add_option("--bkg_cut", type='int', default = 40,  help="What classifier percentile to use to define bkg-rich region")

parser.add_option("--filt_sig", default = False, action = "store_true", help="Reduce the amount of signal in the dataset")
parser.add_option("-s", "--sig_frac", type = 'float', default = 0.01,  help="Reduce signal to this amount (default is 0.01)")
parser.add_option("--no_end_str", default = False, action = "store_true",  help="Don't do automatic end string based on signal frac")


(options, args) = parser.parse_args()


plot_dir = options.plot_dir
model_dir = options.model_dir
extra_label = options.extra_label




#################################################################
end_str = ".h5"
if(not options.no_end_str):
    if(options.filt_sig == False): end_str = "_9p.h5"
    elif(options.sig_frac == 0.01): end_str = "_1p.h5"
    elif(options.sig_frac == 0.003): end_str = "_03p.h5"
    elif(options.sig_frac == 0.001): end_str = "_01p.h5"
print("End string is ", end_str)

if(options.use_dense): network_type = "dense"
else: network_type = "CNN"

if(options.tnt_iter == 0):
    if(not options.start_cwbh):
        print("In interation 0. Using autoencoder as starting network")
        labeler_name = "auto_encoder" + end_str
    else :
        print("In interation 0. Using cwola bumphunt as starting network")
        labeler_name = "cwbh_dense" + end_str
else:
    labeler_name = "TNT" + str(options.tnt_iter - 1) + "_" + network_type + "_" + extra_label  +  end_str

model_name = "TNT" + str(options.tnt_iter) + "_" + network_type + "_" + extra_label + end_str

if(options.retrain):
    model_start = model_name
else:
    model_start = options.model_start

plot_prefix = "TNT" + str(options.tnt_iter) + "_" + network_type + "_" + extra_label 

#start with different data not to overlap training sets
data_start = options.evt_offset + options.num_data * (options.tnt_iter + 1)
print("TNT iter is ", options.tnt_iter)
print("Will train using %i events, starting at event %i" % (options.num_data, data_start))



npix = 40
input_shape = (npix, npix)
val_frac = 0.1
test_frac = 0.0
batch_size = 200




hf_in = h5py.File(options.fin, "r")

j1_images = hf_in['j1_images'][data_start:data_start + options.num_data]
j1_images = np.expand_dims(j1_images, axis=-1)
j2_images = hf_in['j2_images'][data_start:data_start + options.num_data]
j2_images = np.expand_dims(j2_images, axis=-1)
jet_infos = hf_in['jet_infos'][data_start:data_start + options.num_data]
Y = np.array(jet_infos[:,0]).reshape(-1) #is signal bit is first bit of info
j1_4vec = jet_infos[:,1:5]
j2_4vec = jet_infos[:,5:9]
mjj = jet_infos[:,9]
mjj_window = ((mjj > options.mjj_low) & (mjj < options.mjj_high)).reshape(-1)

#which images to train on and which to use for labelling
if(options.training_j == 1):
    j_label = "j1_"
    opp_j_label = "j2_"
    train_images = j1_images
    labeling_images = j2_images
    j_pts = jet_infos[:,1] #j1 pt
    print("training classifier for j1 using j2 for labeling")

elif (options.training_j ==2):
    j_label = "j2_"
    opp_j_label = "j1_"
    train_images = j2_images
    labeling_images = j1_images
    j_pts = jet_infos[:,5] #j2 pt
    print("training classifier for j2 using j1 for labeling")
else:
    print("Training jet not 1 or 2! Exiting")
    exit(1)


if(options.use_dense):
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

else:
    X = train_images
    L = labeling_images

    X = standardize(*zero_center(X))[0]
    L = standardize(*zero_center(L))[0]


    





model_save_path = model_dir+ j_label+ model_name
f_labeler = model_dir + opp_j_label + labeler_name

labeler_plot = plot_dir+ opp_j_label +  plot_prefix + "_labeler_regions.png"
pt_plot = plot_dir + j_label + plot_prefix + "pt_dists.png"
pt_rw_plot = plot_dir + j_label + plot_prefix + "pt_rw_dists.png"
training_plot = plot_dir + j_label + plot_prefix + "training_history.png"


if(options.filt_sig):

    print("Filtering sig to be %.3f" % options.sig_frac)
#force signal to be a given fraction of total events because input dataset signal is too big (10%)
    mask = get_signal_mask(Y, options.sig_frac)

    X = X[mask]
    Y = Y[mask]
    L = L[mask]
    mjj = mjj[mask]
    j_pts = j_pts[mask]



print("Loading labeling model from %s " % f_labeler)
labeler = load_model(f_labeler)

L_pred = labeler.predict(L, batch_size = 500)

print("Using model %s as labeler \n" % labeler_name)
if("auto_encoder" in labeler_name):
    L_labeler_scores =  np.mean(keras.losses.mean_squared_error(L_pred, L), axis=(1,2))
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



(X, Y_true, jet_pts, mjj), Y_lab = sample_split(X, Y, j_pts, mjj, cut_var = L_labeler_scores, sig_high = True, sig_cut = sig_region_cut, bkg_cut = bkg_region_cut)


print_signal_fractions(Y_true, Y_lab)
if(options.mjj_cut):
    #remove sig-like events not in mjj window
    print("Requiring mjj window from %.0f to %.0f \n" % (options.mjj_low, options.mjj_high))
    mjj_window_sig = ((mjj > options.mjj_low) & (mjj < options.mjj_high)).reshape(-1)
    if(options.mjj_cut_sideband):
        #use sidebands for bkg samples
        window_size = 300.
        print("Requiring sideband window of size %.0f around signal window for backgrounds \n" % window_size)
        mjj_window_bkg = ((mjj > (options.mjj_low - window_size)) &  (mjj < (options.mjj_high + window_size)))
        keep_events =  (mjj_window_bkg & (Y_lab < 0.1)) | (mjj_window_sig & (Y_lab > 0.9))
    else:
        print(mjj_window_sig.shape, Y_lab.shape)
        keep_events =  (Y_lab < 0.1) | (mjj_window_sig & (Y_lab > 0.9))


    X = X[keep_events]
    Y_lab = Y_lab[keep_events]
    Y_true = Y_true[keep_events]
    jet_pts = jet_pts[keep_events]


    print("New sig fracs are:  ")
    print_signal_fractions(Y_true, Y_lab)

(X_train, X_val, X_test, 
        jet_pts_train, jet_pts_val, jet_pts_test,
        Y_true_train, Y_true_val, Y_true_test,
        Y_lab_train, Y_lab_val, Y_lab_test) = data_split(X, jet_pts, Y_true, Y_lab, val=val_frac, test=test_frac, shuffle = True)

evt_weights = np.ones(X_train.shape[0])



print(Y_lab_train)


if(options.reweight):
    print("Doing reweighting based on jet pt")
    sr_pts = jet_pts_train[Y_lab_train[:,0] > 0.9]
    br_pts = jet_pts_train[Y_lab_train[:,0] < 0.1]
    labels = ['Signal', 'Background']
    colors = ['b', 'r']
    n_pt_bins = 20
    bins, ratio = make_ratio_histogram([sr_pts, br_pts], labels, colors, 'jet pt (GeV)', "Sig vs. Bkg Pt distribution", n_pt_bins,
                    normalize=True, save = True, fname=pt_plot)
    rw_idxs = np.digitize(jet_pts_train, bins = bins) - 1
    
    rw_idxs = np.clip(rw_idxs, 0, len(ratio)-1) #handle overflows
    rw_vals = ratio[rw_idxs]
    #don't reweight signal region
    rw_vals[Y_lab_train[:,0] > 0.9] = 1.

    sr_wgts = rw_vals[Y_lab_train[:,0] > 0.9] 
    br_wgts = rw_vals[Y_lab_train[:,0] < 0.1] 
    bins, ratio = make_ratio_histogram([sr_pts, br_pts], labels, colors, 'jet pt (GeV)', "Reweighted Sig vs. Bkg Pt distribution", n_pt_bins,
                    normalize=True, weights = [sr_wgts, br_wgts], save = True, fname=pt_rw_plot)

    evt_weights = rw_vals







myoptimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.8, beta_2=0.99, epsilon=1e-08, decay=0.0005)

if(model_start == ""):
    print("Creating new model ")
    if(options.use_dense): my_model = dense_net(X_train.shape[1])
    else: my_model = CNN(X_train[0].shape)

    my_model.summary()
    my_model.compile(optimizer=myoptimizer,loss='binary_crossentropy',
              metrics = [keras.metrics.AUC()]
            )
else:
    print("Starting with model from %s " % model_start)
    my_model = load_model(model_dir + j_label + model_start)


early_stop = keras.callbacks.EarlyStopping(monitor='val_auc', min_delta=0, patience=10, verbose=1, mode='max', baseline=None, restore_best_weights=True)

additional_val = AdditionalValidationSets([(X_val, Y_true_val, "Val_true_sig")], batch_size = 500)

cbs = [keras.callbacks.History()] 

#checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath=model_dir + j_label + "ckpt{epoch:02d}_"+model_name, 
#       monitor='val_auc', verbose=1, save_best_only=False, save_weights_only=False, mode='max', period = 5)
#if(checkpoint): cbs.append(checkpoint_cb)
cbs.append(additional_val)




# train model
history = my_model.fit(X_train, Y_lab_train,
          epochs=options.num_epoch,
          batch_size=batch_size,
          validation_data=(X_val, Y_lab_val),
          sample_weight = evt_weights,
          callbacks = cbs,
          verbose=1)

print("Saving model to : ", model_save_path)
my_model.save(model_save_path)
plot_training(history.history, fname = training_plot)
