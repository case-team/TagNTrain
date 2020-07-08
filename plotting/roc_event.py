import sys
sys.path.append('..')
from utils.TrainingUtils import *
import h5py

fin = "../data/BB_images_v1.h5"
#f_sig = "../data/BulkGravToZZToZhadZhad_narrow_M-2500.h5" #1
#f_sig = "../data/WprimeToWZToWhadZhad_narrow_M-3500.h5" #2
f_sig = "../data/WkkToWRadionToWWW_M2500-R0-08.h5" #3
f_bkg = "../data/QCD_only.h5"
plot_name = "s3_evt_roc.png"
sig_idx = 3
m_low = 2250.
m_high = 2750.


single_file = False

plot_dir = "../plots/BB1_test_june/"
model_dir = "../models/BB1_test_june/"


#model types: 0 CNN (one jet), 1 auto encoder, 2 dense (one jet), 3 CNN (both jets), 4 dense (both jets)
f_models = ["jj_supervised_s%i.h5", "autoencoder.h5", "autoencoder_s%i.h5", "jj_cwola_hunting_s%i.h5", "cwola_hunting_s%i.h5", "TNT0_s%i.h5", "TNT0_s%i_v2.h5", ]
labels = ["Supervised", "Autoencoder (train all)", "Autoencoder (train sidebands)", "CWoLa Hunting", "CWoLa Hunting (one-jet)", "TNT (AE)", "TNT (CWoLa)" ]
model_type = [3, 1, 1, 3,  0,0,0] 
colors = ["g", "b", "r", "gray", "purple", "pink", "orange", "m", "skyblue", "yellow"]

n_points = 200.

logy= True


keys = ["j1_images", "j2_images", "jj_images"]

if(single_file):
    num_data = 400000
    data_start = 20000000
    data = DataReader(fin, signal_idx = sig_idx, start = data_start, stop = data_start + num_data, keys = keys, m_low = m_low, m_high = m_high )
    data.read()
    j1_images = data['j1_images']
    j2_images = data['j2_images']
    jj_images = data['jj_images']
    Y = data['label']

else:
    bkg_start = 1000000
    n_bkg = 400000
    sig_start = 20000
    sig_stop = -1
    d_bkg = DataReader(f_bkg, keys = keys, signal_idx = -1, start = bkg_start, stop = bkg_start + n_bkg, m_low = m_low, m_high = m_high )
    d_bkg.read()
    d_sig = DataReader(f_sig, keys = keys, signal_idx = -1, start = sig_start, stop = sig_stop, m_low = m_low, m_high = m_high )
    d_sig.read()

    j1_im_bkg = d_bkg['j1_images']
    j1_im_sig = d_sig['j1_images']
    j1_images = np.concatenate((j1_im_bkg, j1_im_sig), axis = 0)

    j2_im_bkg = d_bkg['j2_images']
    j2_im_sig = d_sig['j2_images']
    j2_images = np.concatenate((j2_im_bkg, j2_im_sig), axis = 0)

    jj_im_bkg = d_bkg['jj_images']
    jj_im_sig = d_sig['jj_images']
    jj_images = np.concatenate((jj_im_bkg, jj_im_sig), axis = 0)

    Y = np.concatenate((np.zeros(j1_im_bkg.shape[0], dtype=np.int8), np.ones(j1_im_sig.shape[0], dtype=np.int8)))

j1_dense_inputs = None
j2_dense_inputs = None

print(j1_images.shape)


# reading images
#filter signal

sig_effs = []
bkg_effs = []
aucs = []
for idx,f in enumerate(f_models):
    if('%' in f): f = f % sig_idx
    print(idx, f, model_type[idx])
    if(model_type[idx]  <= 2): #classifier on each jet
        if(len(f) != 2):
            j1_model = tf.keras.models.load_model(model_dir + "j1_" + f)
            j2_model = tf.keras.models.load_model(model_dir + "j2_" + f)
        else:
            j1_model = tf.keras.models.load_model(model_dir + f[0])
            j2_model = tf.keras.models.load_model(model_dir + f[1])

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
            jj_model = tf.keras.models.load_model(model_dir + f)
            scores = jj_model.predict(jj_images, batch_size = 1000).reshape(-1)
            bkg_eff, sig_eff, thresholds_cwola = roc_curve(Y, scores)
            #print('bkg eff 10% ',f,np.percentile(scores,10),len(Y[(scores > np.percentile(scores,10)) & (Y==0)])/len(Y[Y==0]),bkg_eff)
            sig_effs.append(sig_eff)
            bkg_effs.append(bkg_eff)
            aucs.append(auc(bkg_eff, sig_eff))


        if(model_type[idx] == 4): #Dense
            jj_model = tf.keras.models.load_model(model_dir + "jj_" + f)
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


