import sys
sys.path.append('..')
from utils.TrainingUtils import *
#from rootconvert import to_root
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential, load_model
from energyflow.utils import data_split, pixelate, standardize, to_categorical, zero_center
import h5py

fin = "/home//oz.amram/LHC_Olympics2020/data/jet_images_blackbox1.h5"
fin_cwola = "/home//oz.amram/LHC_Olympics2020/data/events_cwbh_blackbox1.h5"

model_dir = "/home//oz.amram/LHC_Olympics2020/training/tag_and_train/models_bb1/"

#model types: 0 CNN (one jet), 1 auto encoder, 2 dense (one jet), 3 CNN (both jets), 4 dense (both jets)
model_name = "TNT1_CNN_mjj3500_v2.h5"
model_type = 0

flab = model_name.replace('.h5','')

fout_name = 'h5_files/bb1_mjj3500_TNT1_v2.h5'

filt_sig = False
sig_frac = 0.01

num_data = 400000
data_start = 200000

use_dense = (model_type == 2 or model_type == 4)
# reading images
hf_in = h5py.File(fin, "r")

j1_images_raw = hf_in['j1_images'][data_start:data_start + num_data]
j1_images = np.expand_dims(j1_images_raw, axis=-1)
j1_images = standardize(*zero_center(j1_images))[0]
j2_images_raw = hf_in['j2_images'][data_start:data_start + num_data]
j2_images = np.expand_dims(j2_images_raw, axis=-1)
j2_images = standardize(*zero_center(j2_images))[0]

jet_infos = hf_in['jet_infos'][data_start:data_start + num_data]
Y = jet_infos[:,0] #is signal bit is first bit of info
j1_Ms = np.maximum(jet_infos[:,4], jet_infos[:,8])
j2_Ms = np.minimum(jet_infos[:,4], jet_infos[:,8])

dijet_mass = jet_infos[:,9]

new_df_unmask = pd.DataFrame()
new_df_unmask['is_signal'] = Y
new_df_unmask['Mjj'] = dijet_mass
new_df_unmask['Mj1'] = j1_Ms
new_df_unmask['Mj2'] = j2_Ms


if(use_dense):
    pd_events = pd.read_hdf(dense_fin)
    pd_events = clean_events(pd_events)
    idx1_start = 2
    idx1_end = 8
    idx2_end = 14
    j1_dense_inputs = pd_events.iloc[data_start:data_start + num_data, idx1_start:idx1_end].values
    j2_dense_inputs = pd_events.iloc[data_start:data_start + num_data, idx1_end:idx2_end].values


j1_images = np.expand_dims(j1_images_raw, axis=-1)
j1_images = standardize(*zero_center(j1_images, np.zeros_like(j1_images)))[0]
j2_images = np.expand_dims(j2_images_raw, axis=-1)
j2_images = standardize(*zero_center(j2_images, np.zeros_like(j2_images)))[0]

batch_size = 1000

if(model_type <= 2):
    j1_model = load_model(model_dir + "j1_" + model_name)
    j2_model = load_model(model_dir + "j2_" + model_name)
    if(model_type == 0):
        print("computing scores")
        j1_scores = j1_model.predict(j1_images, batch_size = batch_size)
        j2_scores = j2_model.predict(j2_images, batch_size = batch_size)
    elif(model_type ==1):
        j1_reco_images = j1_model.predict(j1_images, batch_size = batch_size)
        j2_reco_images = j2_model.predict(j2_images, batch_size = batch_size)
        j1_scores =  np.mean(keras.losses.mean_squared_error(j1_reco_images, j1_images), axis=(1,2)).reshape(-1)
        j2_scores =  np.mean(keras.losses.mean_squared_error(j2_reco_images, j2_images), axis=(1,2)).reshape(-1)
    elif(model_type == 2):
        j1_scores = j1_model.predict(j1_dense_inputs, batch_size = batch_size)
        j2_scores = j2_model.predict(j2_dense_inputs, batch_size = batch_size)

    j1_scores = j1_scores.reshape(-1)
    j2_scores = j2_scores.reshape(-1)
    new_df_unmask['j1_score'] = j1_scores
    new_df_unmask['j2_score'] = j2_scores
else:
    threshholds = jj_threshholds
    jj_model = load_model(model_dir + "jj_" + model_name)
    X = np.stack((j1_images_raw,j2_images_raw), axis = -1)
    X = standardize(*zero_center(X))[0]
    jj_scores = jj_model.predict(X, batch_size = batch_size).reshape(-1)

    new_df_unmask['score_%s'%flab] = jj_scores


if(filt_sig):
    sig_mask = get_signal_mask(Y, sig_frac)
    newdf = new_df_unmask[sig_mask]
else:
    newdf = new_df_unmask

print(jet_infos)

print(newdf)
print(newdf.columns.values,len(newdf.columns.values))
        
h5File = h5py.File(fout_name,'w')
h5File.create_dataset('test', data=newdf.as_matrix(),  compression='lzf')
h5File.close()
