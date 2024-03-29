import sys
sys.path.append('..')
from utils.TrainingUtils import *
from optparse import OptionParser
from optparse import OptionGroup




parser = OptionParser()
parser = OptionParser(usage="usage: %prog analyzer outputfile [options] \nrun with --help to get list of options")
parser.add_option("-i", "--fin", default='../data/jet_images.h5', help="Input file for training.")
parser.add_option("--plot_dir", default='../plots/', help="Directory to output plots")
parser.add_option("-o", "--model_name", default='vae_test', help="What to name the model")
parser.add_option("--num_epoch", type = 'int', default=100, help="How many epochs to train for")
parser.add_option("--data_start", type='int', default=0, help="Starting event")
parser.add_option("--num_data", type='int', default=-1, help="How many events to use for training (before filtering)")
parser.add_option("--batch_start", type='int', default=-1, help="Train over multiple batches of dataset. Starting batch")
parser.add_option("--batch_stop", type='int', default=-1, help="Train over multiple batches of dataset. Stopping batch (inclusive)")

parser.add_option("--model_start", default="", help="Starting point for model (empty string for new model)")
parser.add_option("--no_mjj_cut", default = False, action = "store_true", help="Don't require a mass window")
parser.add_option("--mjj_low", type='int', default = 3300,  help="Low mjj cut value")
parser.add_option("--mjj_high", type='int', default = 3700, help="High mjj cut value")
parser.add_option("--norm_img", default = '', help="h5 file with avg and std dev of image")

parser.add_option("-j", "--training_j", type ='int', default = 1, help="Which jet to make a classifier for (1 or 2)")

parser.add_option("--sig_idx", type = 'int', default = 1,  help="What index of signal to use")
parser.add_option("-s", "--sig_frac", type = 'float', default = 0.01,  help="Reduce signal to this amount (default is 0.01)")

(options, args) = parser.parse_args()


val_frac = 0.1
batch_size = 256



#which images to train on and which to use for labelling
if(options.training_j == 1):
    j_label = "j1_"
    x_key = 'j1_images'
    cnn_shape = (32,32,1)
    print("training autoencoder for j1")

elif (options.training_j ==2):
    j_label = "j2_"
    x_key = 'j2_images'
    cnn_shape = (32,32,1)
    print("training autoencoder for j2")
else:
    print("Training jet not 1 or 2! Exiting")
    exit(1)

keep_low = -1.
keep_high = -1.
if(not options.no_mjj_cut):
    window_size = (options.mjj_high - options.mjj_low)/2.
    keep_low = options.mjj_low - window_size
    keep_high = options.mjj_high + window_size
    print("Requiring mjj window from %.0f to %.0f \n" % (options.mjj_low, options.mjj_high))


import time
keys  = [x_key, 'mjj']
t1 = time.time()
data = DataReader(options.fin, keys = keys, signal_idx = options.sig_idx, sig_frac = options.sig_frac, start = options.data_start, stop = options.data_start + options.num_data, 
         val_frac = val_frac, m_high = keep_high, m_low = keep_low, batch_start = options.batch_start, batch_stop = options.batch_stop, norm_img = options.norm_img)
data.read()
t2 = time.time()
print("load time  %s " % (t2 -t1))

mjj = data['mjj']
mjj_cut = (mjj < options.mjj_low) | (mjj > options.mjj_high)
data.apply_mask(mjj_cut)

if(val_frac > 0.):
    val_mjj = data['val_mjj']
    val_mjj_cut = (val_mjj < options.mjj_low) | (val_mjj > options.mjj_high)
    data.apply_mask(val_mjj_cut, to_training = False)

mjj_cut = data['mjj']


print("Signal frac is %.3f \n" % (np.mean(data['label'])))

if(options.model_start == ""):
    print("Creating new model ")
    my_model = VAE(0, input_shape = cnn_shape, model_dir = options.model_name)
    my_model.build()
    my_model.model.summary()
else:
    print("Loading model not yet implemented" % model_start)
    my_model = VAE(0, input_shape = cnn_shape, model_dir = options.model_name)
    my_model.load()
    my_model.model.summary()
    exit(1)



# train model
t_data = data.gen(x_key,x_key, batch_size = batch_size)
v_data = None
if(val_frac > 0.): 
    v_data = data.gen('val_'+x_key,'val_'+x_key, batch_size = batch_size)

print("Will train on %i events, validate on %i events" % (data.nTrain, data.nVal))



history = my_model.fit(t_data, 
        epochs = options.num_epoch, 
        validation_data = v_data,
        verbose = 2 )


print("Saving model to : ", options.model_name)
os.system("mkdir %s" % options.model_name)
my_model.save_model()
data.cleanup()
