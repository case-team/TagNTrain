import sys
sys.path.append('..')
from utils.TrainingUtils import *
from optparse import OptionParser
from optparse import OptionGroup


parser = OptionParser()
parser = OptionParser(usage="usage: %prog analyzer outputfile [options] \nrun with --help to get list of options")
parser.add_option("-i", "--fin", default='../data/jet_images.h5', help="Input file for training.")
parser.add_option("--plot_dir", default='../plots/', help="Directory to output plots")
parser.add_option("-o", "--model_name", default='supervised_CNN.h5', help="What to name the model")
parser.add_option("--num_epoch", type = 'int', default=30, help="How many epochs to train for")
parser.add_option("--data_start", type = 'int', default=0, help="What event to start with")
parser.add_option("--num_data", type='int', default=200000, help="How many events to use for training (before filtering)")

parser.add_option("--large", default = False, action = "store_true", help="Use larger NN archetecture")
parser.add_option("--use_one", default = False, action = "store_true", help="Make a classifier for one jet instead of both")
parser.add_option("-j", "--training_j", type ='int', default = 1, help="Which jet to make a classifier for (1 or 2)")
parser.add_option("--sig_idx", type = 'int', default = 1,  help="What index of signal to use")

(options, args) = parser.parse_args()

val_frac = 0.1
batch_size = 256


if(not options.use_one):
    j_label = "jj_"
    x_key = 'jj_images'
    cnn_shape = (32,32,2)
    print("training classifier for both jets")
elif(options.training_j == 1):
    j_label = "j1_"
    x_key = 'j1_images'
    cnn_shape = (32,32,1)
    print("training classifier for j1")

elif (options.training_j ==2):
    j_label = "j2_"
    x_key = 'j2_images'
    cnn_shape = (32,32,1)
    print("training classifier for j2")
else:
    print("Training jet not 1 or 2! Exiting")
    exit(1)


import time
keys = [x_key]
t1 = time.time()
data = DataReader(options.fin, keys = keys, signal_idx = options.sig_idx, start = options.data_start, stop = options.data_start + options.num_data, 
        val_frac = val_frac )
data.read()
t2 = time.time()
print("load time  %s " % (t2 -t1))

print("Signal frac is %.3f \n" % (np.mean(data['label'])))

if(options.large):
    my_model = CNN_large(cnn_shape)
else:
    my_model = CNN(cnn_shape)
my_model.summary()

myoptimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.8, beta_2=0.99, epsilon=1e-08, decay=0.0005)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=5, verbose=1, mode='min')
#roc = RocCallback(training_data=(X_train, Y_train), validation_data=(X_val, Y_val))

cbs = [tf.keras.callbacks.History(), early_stop]

my_model.compile(optimizer=myoptimizer,loss='binary_crossentropy',
          metrics = ['accuracy'],
        )


t_data = data.gen(x_key,'label', batch_size = batch_size)
v_data = None
if(val_frac > 0.): 
    v_data = data.gen('val_'+x_key,'val_label', batch_size = batch_size)

print("Will train on %i events, validate on %i events" % (data.nTrain, data.nVal))

history = my_model.fit(t_data, 
        epochs = options.num_epoch, 
        validation_data = v_data,
        callbacks = cbs,
        verbose = 2 )

my_model.save(options.model_name)


