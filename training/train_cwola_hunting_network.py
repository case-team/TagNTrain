import sys
sys.path.append('..')
from utils.TrainingUtils import *
from optparse import OptionParser
from optparse import OptionGroup



parser = OptionParser()
parser = OptionParser(usage="usage: %prog analyzer outputfile [options] \nrun with --help to get list of options")
parser.add_option("-i", "--fin", default='../data/jet_images.h5', help="Input file with data for training.")
parser.add_option("--plot_dir", default='../plots/', help="Directory to output plots")
parser.add_option("--model_dir", default='../models/', help="Directory to read in and output models")
parser.add_option("-o", "--model_name", default='cwbh.h5', help="What to name the model")
parser.add_option("--num_epoch", type = 'int', default=30, help="How many epochs to train for")
parser.add_option("--data_start", type = 'int', default=0, help="What event to start with")
parser.add_option("--num_data", type = 'int', default=200000, help="How many events to train on")

parser.add_option("--use_one", default = False, action = "store_true", help="Make a classifier for one jet instead of both")
parser.add_option("-j", "--training_j", type ='int', default = 1, help="Which jet to make a classifier for (1 or 2)")
parser.add_option("--use_dense", default = False, action = "store_true", help="Make a classifier using inputs from cwola hunting paper instead of jet images")
parser.add_option("--mjj_low", type='int', default = 3300,  help="Low mjj cut value")
parser.add_option("--mjj_high", type='int', default = 3700, help="High mjj cut value")

parser.add_option("--large", default = False, action = "store_true", help="Use larger NN archetecture")
parser.add_option("--sig_idx", type = 'int', default = 1,  help="What index of signal to use")
parser.add_option("-s", "--sig_frac", type = 'float', default = -1.,  help="Reduce signal to this amount (< 0 to not filter )")


(options, args) = parser.parse_args()


plot_dir = options.plot_dir
model_dir = options.model_dir




model_name = options.model_name






#################################################################





val_frac = 0.0
batch_size = 256



window_size = (options.mjj_high - options.mjj_low)/2.
keep_low = options.mjj_low - window_size
keep_high = options.mjj_high + window_size


#which images to train on and which to use for labelling
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


if(not options.use_dense):
    import time
    keys = ['mjj']
    keys.append(x_key)
    t1 = time.time()
    data = DataReader(options.fin, keys = keys, signal_idx = options.sig_idx, sig_frac = options.sig_frac, start = options.data_start, stop = options.data_start + options.num_data, 
            m_low = keep_low, m_high = keep_high, val_frac = val_frac )
    data.read()
    data.make_Y_mjj(options.mjj_low, options.mjj_high)
    t2 = time.time()
    print("load time  %s " % (t2 -t1))








print_signal_fractions(data['label'], data['Y_mjj'])






myoptimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.8, beta_2=0.99, epsilon=1e-08, decay=0.0005)
my_model = CNN(cnn_shape)
my_model.summary()
my_model.compile(optimizer=myoptimizer,loss='binary_crossentropy',
        metrics = ['accuracy'])

if(options.model_name == ""):
    f_model = model_dir+ j_label+ "cwbh.h5"
else:
    f_model = options.model_name
print("Will save model to : ", f_model)


#additional_val = AdditionalValidationSets([(X_val, Y_val_true, "Val_true_sig")], batch_size = 500)

#roc = RocCallback(training_data=(X_train, Y_train), validation_data=(X_val, Y_val))
#cbs = [callbacks.History(), additional_val, roc] 





# train model
t_data = data.gen(x_key,'Y_mjj', batch_size = batch_size)
v_data = None
if(val_frac > 0.): 
    v_data = data.gen('val_'+x_key,'val_label', batch_size = batch_size)

print("Will train on %i events, validate on %i events" % (data.nTrain, data.nVal))
#print(np.mean(data['val_label']))
#print(np.mean(data['label']))

history = my_model.fit(t_data, 
        epochs = options.num_epoch, 
        validation_data = v_data,
        #callbacks = cbs,
        verbose = 2 )

plot_training(history.history, plot_dir + j_label + "training_history.png")
print("Saving model to : ", f_model)
my_model.save(f_model)
