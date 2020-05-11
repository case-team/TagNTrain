import sys
sys.path.append('..')
from utils.TrainingUtils import *
#from rootconvert import to_root
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential, load_model
from energyflow.utils import data_split, pixelate, standardize, to_categorical, zero_center
import h5py

fin = "../data/jet_images.h5"
fin_cwola = "../data/events_cwbh_v3.h5"

plot_dir = "../plots/"

model_dir = "../models/"

plot_name = "roc_event_03p.png"

#model types: 0 CNN (one jet), 1 auto encoder, 2 dense (one jet), 3 CNN (both jets), 4 dense (both jets)
#f_models = ["supervised_CNN.h5", "supervised_CNN.h5", "auto_encoder_9p.h5", "cwbh_CNN_9p.h5", "TNT1_CNN_no_mjj_9p.h5", "TNT1_CNN__9p.h5"]
#f_models = ["supervised_CNN.h5", "supervised_CNN.h5", "auto_encoder_1p.h5", "cwbh_CNN_1p.h5", "TNT1_CNN_no_mjj_s10p_1p.h5",  "TNT2_CNN_s10p_1p.h5"]
f_models = ["supervised_CNN.h5", "supervised_CNN.h5", "auto_encoder_03p.h5", "cwbh_CNN_03p.h5", "TNT0_CNN_no_mjj_03p.h5", "TNT1_CNN_s10p_03p.h5"]
#f_models = ["supervised_CNN.h5", "supervised_CNN.h5", "auto_encoder_01p.h5", "cwbh_CNN_01p.h5", "TNT0_CNN_no_mjj_01p.h5", "TNT2_CNN_s10p_01p.h5"]
labels = ["Supervised, both jets","Supervised, separate", "Autoencoders", "CWola hunting", "TNT", "TNT + M$_{jj}$"]
model_type = [3, 0, 1, 3, 0, 0, 0, 0]
colors = ["g", "gray", "b", "r","m","c", "skyblue", "yellow"]


sig_frac = 0.01

num_data = 100000
data_start = 1000000

n_points = 200.

make_h5 = False
logy= True

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
j1_4vec = jet_infos[:,1:5]
j2_4vec = jet_infos[:,5:]

# reading cwola
idx1_start = 2
idx1_end = 8
idx2_end = 14
pd_events = pd.read_hdf(fin_cwola)
pd_events = clean_events(pd_events)
X_test = pd_events.iloc[data_start:data_start + num_data, idx1_start:idx2_end].values
y_test = pd_events.iloc[data_start:data_start + num_data, [0]].values

j1_dense_inputs = pd_events.iloc[data_start:data_start + num_data, idx1_start:idx1_end].values
j2_dense_inputs = pd_events.iloc[data_start:data_start + num_data, idx1_end:idx2_end].values

newdf_unmask = pd_events[['is_signal','Mjj','Mj1','Mj2']][data_start:data_start + num_data].copy()
#filter signal
sig_mask = get_signal_mask(y_test, sig_frac)
if(make_h5):
    newdf = newdf_unmask[sig_mask]
    #print(len(newdf_unmask['Mjj']),len(newdf['Mjj']))

sig_effs = []
bkg_effs = []
aucs = []
for idx,f in enumerate(f_models):
    print(idx, f, model_type[idx])
    flab = f.replace('.h5','')
    if(model_type[idx]  <= 2): #classifier on each jet
        j1_model = load_model(model_dir + "j1_" + f)
        j2_model = load_model(model_dir + "j2_" + f)
        if(model_type[idx] == 0):  #CNN
            j1_score = j1_model.predict(j1_images, batch_size = 500)
            j2_score = j2_model.predict(j2_images, batch_size = 500)
        elif(model_type[idx] == 2): #dense
            j1_score = j1_model.predict(j1_dense_inputs, batch_size = 500)
            j2_score = j2_model.predict(j2_dense_inputs, batch_size = 500)

        else: #autoencoder
            j1_reco_images = j1_model.predict(j1_images, batch_size=500)
            j1_score =  np.mean(keras.losses.mean_squared_error(j1_reco_images, j1_images), axis=(1,2))
            j2_reco_images = j2_model.predict(j2_images, batch_size=500)
            j2_score =  np.mean(keras.losses.mean_squared_error(j2_reco_images, j2_images), axis=(1,2))
        j1_score = j1_score.reshape(-1)
        j2_score = j2_score.reshape(-1)
        sig_eff = [len(Y[(j1_score > np.percentile(j1_score,i)) & (j2_score > np.percentile(j2_score,i)) & (Y==1)])/len(Y[Y==1]) for i in np.arange(0.,100., 100./n_points)]
        bkg_eff = [len(Y[(j1_score > np.percentile(j1_score,i)) & (j2_score > np.percentile(j2_score,i)) & (Y==0)])/len(Y[Y==0]) for i in np.arange(0.,100., 100./n_points)]
        #print('bkg eff 10% ',f,np.percentile(j1_score,10),np.percentile(j2_score,10),bkg_eff[11])
        sig_effs.append(sig_eff)
        bkg_effs.append(bkg_eff)
        aucs.append(auc(bkg_eff, sig_eff))

        if(make_h5):
            newdf['j1_score_%s'%flab] = j1_score[sig_mask]
            newdf['j2_score_%s'%flab] = j2_score[sig_mask]

    else: #classifier for both jets
        if(model_type[idx] == 3): #CNN
            jj_model = load_model(model_dir + "jj_" + f)
            X = np.stack((j1_images_raw,j2_images_raw), axis = -1)
            X = standardize(*zero_center(X))[0]
            scores = jj_model.predict(X, batch_size = 1000).reshape(-1)
            bkg_eff, sig_eff, thresholds_cwola = roc_curve(Y, scores)
            #print('bkg eff 10% ',f,np.percentile(scores,10),len(Y[(scores > np.percentile(scores,10)) & (Y==0)])/len(Y[Y==0]),bkg_eff)
            sig_effs.append(sig_eff)
            bkg_effs.append(bkg_eff)
            aucs.append(auc(bkg_eff, sig_eff))

            if(make_h5): newdf['score_%s'%flab] = scores[sig_mask]

        if(model_type[idx] == 4): #Dense
            jj_model = load_model(model_dir + "jj_" + f)
            X = np.append(j1_dense_inputs, j2_dense_inputs, axis = -1)
            scores = jj_model.predict(X, batch_size = 1000).reshape(-1)
            bkg_eff, sig_eff, thresholds_cwola = roc_curve(Y, scores)
            sig_effs.append(sig_eff)
            bkg_effs.append(bkg_eff)
            aucs.append(auc(bkg_eff, sig_eff))

            if(make_h5): newdf['score_%s'%flab] = scores[sig_mask]
            



fs = 18
fs_leg = 16
plt.figure(figsize=fig_size)
for i in range(len(labels)):
    print(labels[i], aucs[i])
    if(logy):
        temp = np.array(bkg_effs[i])
        #guard against division by 0
        temp = np.clip(temp, 1e-8, 1.)
        ys = 1./temp
    else:
        ys = bkg_effs[i]
    plt.plot(sig_effs[i], ys, lw=2, color=colors[i], label=labels[i] + (" (%.3f)" % aucs[i]))
#plt.plot(fpr_cwola, tpr_cwola, lw=2, color="purple", label="CWOLA = %.3f" %(auc_cwola))
#plt.plot(tnt_bkg_eff, tnt_sig_eff, lw=2, color="r", label="TNT Dense = %.3f"%(auc(tnt_bkg_eff,tnt_sig_eff)))
#plt.plot(sup_bkg_eff, sup_sig_eff, lw=2, color="g", label="Sup. Dense = %.3f"%(auc(sup_bkg_eff,sup_sig_eff)))
#plt.plot(ae_bkg_eff, ae_sig_eff, lw=2, color="b", label="Autoencoders = %.3f"%(auc(ae_bkg_eff,ae_sig_eff)))

#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='random chance')
plt.xlim([0, 1.0])
plt.xlabel('Signal Efficiency', fontsize = fs)
if(logy):
    plt.ylim([1., 1e4])
    plt.yscale('log')
    plt.ylabel('QCD Rejection Rate', fontsize = fs)
else:
    plt.ylim([0, 1.0])
    plt.ylabel('Background Efficiency')
plt.tick_params(axis='x', labelsize=fs_leg)
plt.tick_params(axis='y', labelsize=fs_leg)
plt.legend(loc="upper right", fontsize= fs_leg)
plt.savefig(plot_dir+plot_name)



if(make_h5):
    print(pd_events)
    print(jet_infos)
    print(newdf)
    print(newdf.columns.values,len(newdf.columns.values))
        
if(make_h5):
    h5File = h5py.File('small.h5','w')
    h5File.create_dataset('test', data=newdf.as_matrix(),  compression='lzf')
    h5File.close()
