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
parser.add_option("--data_start", type='int', default=0, help="Starting event")
parser.add_option("--num_data", type='int', default=200000, help="How many events to use for training (before filtering)")
parser.add_option("--model_start", default="", help="Starting point for model (empty string for new model)")

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


import time
keys  = [x_key]
t1 = time.time()
data = DataReader(options.fin, keys = keys, signal_idx = options.sig_idx, sig_frac = options.sig_frac, start = options.data_start, stop = options.data_start + options.num_data, 
         val_frac = val_frac )
data.read()
t2 = time.time()
print("load time  %s " % (t2 -t1))



print("Signal frac is %.3f \n" % (np.mean(data['label'])))

if(options.model_start == ""):
    print("Creating new model ")
    my_model = auto_encoder_large(cnn_shape)
    my_model.summary()
    myoptimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.8, beta_2=0.99, epsilon=1e-08, decay=0.0005)
    my_model.compile(optimizer=myoptimizer,loss= tf.keras.losses.mean_squared_error)
else:
    print("Starting with model from %s " % model_start)
    my_model = load_model(options.model_start)


early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='min', baseline=None)
cbs = [tf.keras.callbacks.History(), early_stop]


# train model
t_data = data.gen(x_key,x_key, batch_size = batch_size)
v_data = None
if(val_frac > 0.): 
    v_data = data.gen('val_'+x_key,'val_'+x_key, batch_size = batch_size)

print("Will train on %i events, validate on %i events" % (data.nTrain, data.nVal))
#print(np.mean(data['val_label']))
#print(np.mean(data['label']))

history = my_model.fit(t_data, 
        epochs = options.num_epoch, 
        validation_data = v_data,
        callbacks = cbs,
        verbose = 2 )


print("Saving model to : ", options.model_name)
my_model.save(options.model_name)
