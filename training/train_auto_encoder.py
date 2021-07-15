import sys
sys.path.append('..')
from utils.TrainingUtils import *
from optparse import OptionParser
from optparse import OptionGroup

parser = input_options()
(options, args) = parser.parse_args()


(options, args) = parser.parse_args()






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

if(options.num_data == -1):
    data_stop = -1
else:
    data_stop = options.data_start + options.num_data

keep_low = -1.
keep_high = -1.
if(not options.no_mjj_cut):
    window_size = (options.mjj_high - options.mjj_low)/2.
    window_frac = window_size / ((options.mjj_high + options.mjj_low)/ 2.)
    window_low_size = window_frac*options.mjj_low / (1 + window_frac)
    window_high_size = window_frac*options.mjj_high / (1 - window_frac)
    keep_low = options.mjj_low - window_low_size
    keep_high = options.mjj_high + window_high_size
    print("Requiring mjj window from %.0f to %.0f \n" % (keep_low, keep_high))



import time
keys  = [x_key, 'mjj']
t1 = time.time()
data = DataReader(options.fin, keys = keys, signal_idx = options.sig_idx, sig_frac = options.sig_frac, start = options.data_start, stop = options.data_start + options.num_data, 
        m_low = keep_low, m_high = keep_high, batch_start = options.batch_start, batch_stop = options.batch_stop , hadronic_only = options.hadronic_only, 
        m_sig = options.mjj_sig, seed = options.BB_seed, eta_cut = options.d_eta, ptsort = options.ptsort, randsort = options.randsort)

data.read()
mjj = data['mjj']
mjj_cut = (mjj < options.mjj_low) | (mjj > options.mjj_high)
data.apply_mask(mjj_cut)

if(options.val_batch_start >0 and options.val_batch_stop > 0):
    do_val = True
    val_data = DataReader(options.fin, keys = keys, signal_idx = options.sig_idx, sig_frac = options.sig_frac, start = options.data_start, stop = options.data_start + options.num_data, 
        m_low = keep_low, m_high = keep_high, batch_start = options.val_batch_start, batch_stop = options.val_batch_stop, 
        m_sig = options.mjj_sig, seed = options.BB_seed, eta_cut = options.d_eta, ptsort = options.ptsort, randsort = options.randsort)
    val_data.read()
    val_mjj = val_data['mjj']
    val_mjj_cut = (val_mjj < options.mjj_low) | (val_mjj > options.mjj_high)
    val_data.apply_mask(val_mjj_cut)
else:
    do_val = False
    val_data = None


t2 = time.time()
print("load time  %s " % (t2 -t1))



mjj_cut = data['mjj']


print("Signal frac is %.3f \n" % (np.mean(data['label'])))



cbs = [tf.keras.callbacks.History()]
myoptimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.8, beta_2=0.99, epsilon=1e-08, decay=0.0005)


# train model
t_data = data.gen(x_key,x_key, batch_size = options.batch_size)
v_data = None
if(do_val): 
    nVal = val_data.nTrain
    v_data = val_data.gen(x_key,x_key, batch_size = options.batch_size)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', baseline=None)
    cbs.append(early_stop)
else:
    nVal = 0

print("Will train on %i events, validate on %i events" % (data.nTrain, nVal))
#print(np.mean(data['val_label']))
#print(np.mean(data['label']))

model_list = []
histories = []

print("Will train %i models" % options.num_models)
for model_idx in range(options.num_models):
    print("Creating model %i" % model_idx)
    model = auto_encoder_large(cnn_shape)
    model.compile(optimizer=myoptimizer,loss= tf.keras.losses.mean_squared_error)
    if(model_idx == 0): model.summary()

    history = model.fit(t_data, 
            epochs = options.num_epoch, 
            validation_data = v_data,
            callbacks = cbs,
            verbose = 2 )


    histories.append(history)
    model_list.append(model)


if(options.num_models == 1):
    best_model = model_list[0]
else:
    min_loss = 9999999
    best_i = -1

    for model_idx in range(options.num_models):
        #preds = model_list[model_idx].predict(val_data[x_key])
        #loss = np.mean(np.mean(np.square(val_data[x_key] - preds), axis = (1,2)).reshape(-1))
        loss = histories[model_idx].history['val_loss'][-1]
        print("Model %i,  val loss %.4f " % (model_idx, loss))
        if(loss < min_loss):
            min_loss = loss
            best_i = model_idx
    print("Selecting model %i " % best_i)
    best_model = model_list[best_i]

print("Saving model to : ", options.model_name)
best_model.save(options.model_name)
data.cleanup()
