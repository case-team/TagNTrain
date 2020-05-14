import sys
sys.path.append('..')
from utils.TrainingUtils import *
#import energyflow as ef
#from energyflow.utils import data_split, pixelate, standardize, to_categorical, zero_center


parser = OptionParser()
parser = OptionParser(usage="usage: %prog analyzer outputfile [options] \nrun with --help to get list of options")
parser.add_option("-i", "--fin", default='../data/jet_images.h5', help="Input file for training.")
parser.add_option("--plot_dir", default='../plots/', help="Directory to output plots")
parser.add_option("-o", "--model_name", default='supervised_CNN.h5', help="What to name the model")
parser.add_option("--num_epoch", type = 'int', default=30, help="How many epochs to train for")
parser.add_option("--num_data", type='int', default=200000, help="How many events to use for training (before filtering)")

parser.add_option("-j", "--training_j", type ='int', default = 1, help="Which jet to make a classifier for (1 or 2)")

signal = 1


val_frac = 0.1
num_epoch = 20
batch_size = 200

use_j1 = False
use_both = True
standardize =False


if(use_both):
    j_label = "jj_"
    print("Training supervised cnn on both jets! label = jj")

elif(use_j1):
    j_label = "j1_"
    print("Training supervised cnn on leading jet! label = j1")
else:
    j_label = "j2_"
    print("Training supervised cnn on sub-leading jet! label = j2")


data = prepare_dataset(options.fin, signal_idx = signal)

if(use_both):
    j1s = data['j1_images'][:options.num_data]
    j2s = data['j2_images'][:options.num_data]
    images = np.stack((j1s,j2s), axis = -1)
else:
    images = data[j_label+'images'][:options.num_data]
    images = np.expand_dims(images, axis=-1)

Y = data['label'][:options.num_data]



(X_train, X_val, Y_train, Y_val) = train_test_split(images, Y, test_size = val_frac)

if(standardize):
    X_train, X_val, X_test = standardize(*zero_center(X_train, X_val, X_test))

print(X_train.shape)


cnn = CNN(X_train[0].shape)
cnn.summary()

myoptimizer = optimizers.Adam(lr=0.001, beta_1=0.8, beta_2=0.99, epsilon=1e-08, decay=0.0005)
#early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto', restore_best_weights=True)
roc = RocCallback(training_data=(X_train, Y_train), validation_data=(X_val, Y_val))
cnn.compile(optimizer=myoptimizer,loss='binary_crossentropy',
          metrics = ['accuracy'],
        )

# train model
history = cnn.fit(X_train, Y_train,
          epochs=options.num_epoch,
          batch_size=batch_size,
          validation_data=(X_val, Y_val),
         callbacks = [roc],
          verbose=2)

# get predictions on test data
#print(Y_predict_test)

#make_roc_curve([Y_predict_test], Y_test,  save = True, fname=plot_dir+ j_label+ "supervised_roc.png")

cnn.save(options.model_name)

