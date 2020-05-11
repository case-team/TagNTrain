import sys
sys.path.append('..')
from utils.TrainingUtils import *
import energyflow as ef
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential, load_model
from energyflow.utils import data_split,  standardize, zero_center
import h5py



fin = "../data/jet_images.h5"
dense_fin = "../data/events_cwbh_v3.h5"
plot_dir = "../plots/"
model_dir = "../models/"
j_label = "j2_"
plot_name = "roc_cmp_1p.png"

num_data = 100000
data_start = 1000000


#model type: 0 is CNN, 1 is autoencoder, 2 is dense network

#f_models = ["cwbh_dense_1p.h5", "supervised_dense.h5", "auto_encoder_1p.h5", "dense_TNT_network0_1p.h5", "dense_TNT_network1_1p.h5", "dense_TNT_network2_1p.h5"]
#labels = ["CWBH Dense", "Supervised Dense", "Auto Encoder", "TNT Dense Iter #1", "TNT Dense Iter #2", "TNT Dense Iter #3"]
#model_type = [2, 2, 1, 2, 2, 2]
#f_models = ["supervised_CNN.h5", "auto_encoder_1p.h5", "TNT0_CNN__9p.h5",  "TNT1_CNN__9p.h5",
       #"TNT0_CNN_no_mjj_9p.h5", "TNT1_CNN_no_mjj_9p.h5"]
#labels = ["Supervised CNN", "Auto Encoder","TNT + Mjj CNN Iter #0",  "TNT + Mjj CNN Iter #1", 
       #"TNT CNN Iter #0", "TNT CNN Iter #1"]

f_models = ["supervised_CNN.h5", "auto_encoder_1p.h5", "TNT0_CNN__1p.h5",  "TNT1_CNN__1p.h5", "TNT2_CNN__1p.h5", "TNT0_CNN_mjj_sb_1p.h5", 
        "TNT1_CNN_mjj_sb_1p.h5", "TNT1_CNN_s10p_1p.h5", "TNT1_CNN_s5p_1p.h5"]
labels = ["Supervised CNN", "Auto Encoder","TNT0 + Mjj", "TNT1 + Mjj", "TNT2 + Mjj",  "TNT0 mjj_sb", "TNT1 mjj_sb",
        "TNT + Mjj 10p",  "TNT + Mjj 5p"]
model_type = [0, 1, 0, 0, 0, 0, 0, 0,0,0]
colors = ["g", "b", "r", "gray", "purple", "pink", "orange", "m", "skyblue", "yellow"]



hf_in = h5py.File(fin, "r")

images = hf_in[j_label+'images'][data_start:data_start + num_data]
images = np.expand_dims(images, axis=-1)
images = standardize(*zero_center(images, np.zeros_like(images)))[0]
jet_infos = hf_in['jet_infos'][data_start:data_start + num_data]
Y = jet_infos[:,0] #is signal bit is first bit of info

pd_events = pd.read_hdf(dense_fin)
pd_events = clean_events(pd_events)
idx1_start = 2
idx1_end = 8
idx2_start = 8 
idx2_end = 14
if(j_label == "j1_"):
    dense_inputs = pd_events.iloc[data_start:data_start + num_data, idx1_start:idx1_end].values
else:
    dense_inputs = pd_events.iloc[data_start:data_start + num_data, idx2_start:idx2_end].values

sig_events = (Y > 0.9).reshape(-1)
bkg_events = (Y < 0.1).reshape(-1)
hist_labels = ['Background', 'Signal']
hist_colors = ['b', 'r']

model_scores = []
for idx,f in enumerate(f_models):
    model = load_model(model_dir + j_label + f)
    print(idx, f, model_type[idx])
    if(model_type[idx] == 0): scores = model.predict(images, batch_size = 500)
    elif(model_type[idx] == 2): scores = model.predict(dense_inputs, batch_size = 500)
    else:
        reco_images = model.predict(images, batch_size=500)
        scores =  np.mean(keras.losses.mean_squared_error(reco_images, images), axis=(1,2))
    scores = scores.reshape(-1)

    hist_scores = [scores[bkg_events], scores[sig_events]]
    make_histogram(hist_scores, hist_labels, hist_colors, 'Labeler Score', "", 100,
            normalize = True, save = True, fname = plot_dir + j_label + f[:-2] + "png")
    model_scores.append(scores)


roc_plt_name  = plot_dir+ j_label + plot_name
make_roc_curve(model_scores, Y, labels = labels, colors = colors, save = True, logy=True, fname=roc_plt_name)
