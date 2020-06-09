import sys
sys.path.append('..')
from utils.TrainingUtils import *
#from rootconvert import to_root
import h5py

fin = "../data/BB_test_Wprime.h5"

plot_dir = "../plots/"

model_dir = "../models/Wprime_test_may13/"

plot_name = "roc_Wprime_test.png"

#model types: 0 CNN (one jet), 1 auto encoder, 2 dense (one jet), 3 CNN (both jets), 4 dense (both jets)
f_models = ["autoencoder.h5", "TNT0.h5"]
labels = ["Initial Classifier (autoencoder) ", "Tag N' Train Classifier"]
model_type = [1, 0] 
colors = ["g", "gray", "b", "r","m","c", "skyblue", "yellow"]


#sig_frac = 0.01
sig_frac = -1.
signal = 1

num_data = 40000
data_start = 400000

n_points = 200.

logy= True


data = DataReader(fin, signal_idx = signal, sig_frac = sig_frac, start = data_start, stop = data_start + num_data )
data.read()
j1_images = data['j1_images']
j2_images = data['j2_images']
Y = data['label']
j1_dense_inputs = None
j2_dense_inputs = None

print(j1_images.shape)


# reading images
#filter signal

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
            j1_score =  np.mean(np.square(j1_reco_images - j1_images), axis=(1,2))
            j2_reco_images = j2_model.predict(j2_images, batch_size=500)
            j2_score =  np.mean(np.square(j2_reco_images -  j2_images), axis=(1,2))
        j1_score = j1_score.reshape(-1)
        j2_score = j2_score.reshape(-1)
        Y = Y.reshape(-1)
        sig_eff = [len(Y[(j1_score > np.percentile(j1_score,i)) & (j2_score > np.percentile(j2_score,i)) & (Y==1)])/len(Y[Y==1]) for i in np.arange(0.,100., 100./n_points)]
        bkg_eff = [len(Y[(j1_score > np.percentile(j1_score,i)) & (j2_score > np.percentile(j2_score,i)) & (Y==0)])/len(Y[Y==0]) for i in np.arange(0.,100., 100./n_points)]
        #print('bkg eff 10% ',f,np.percentile(j1_score,10),np.percentile(j2_score,10),bkg_eff[11])
        sig_effs.append(sig_eff)
        bkg_effs.append(bkg_eff)
        aucs.append(auc(bkg_eff, sig_eff))


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


        if(model_type[idx] == 4): #Dense
            jj_model = load_model(model_dir + "jj_" + f)
            X = np.append(j1_dense_inputs, j2_dense_inputs, axis = -1)
            scores = jj_model.predict(X, batch_size = 1000).reshape(-1)
            bkg_eff, sig_eff, thresholds_cwola = roc_curve(Y, scores)
            sig_effs.append(sig_eff)
            bkg_effs.append(bkg_eff)
            aucs.append(auc(bkg_eff, sig_eff))

            



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
    plt.plot(sig_effs[i], ys, lw=2, color=colors[i], label=labels[i] + (" (AUC = %.3f)" % aucs[i]))
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


