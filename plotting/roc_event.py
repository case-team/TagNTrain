import sys
sys.path.append('..')
from utils.TrainingUtils import *
import h5py
import time

parser = input_options()
options = parser.parse_args()

roc_plot_name = "event_roc_%s.png" %( options.label)
sic_plot_name = "event_sic_%s.png" % (options.label)

compute_mjj_window(options)
options.keep_mlow = options.mjj_low
options.keep_mhigh = options.mjj_high

options.no_minor_bkgs = True
options.hadronic_only = True
options.deta = 1.3


sic_max = 10

#model_dir = "../models/batch_size_test/"
#model_dir = "../models/AEs/"

model_dir = "../runs/TNT_X3000_Y170_Yp170_feb16/spb3.0/"
plot_dir = "../runs/TNT_X3000_Y170_Yp170_feb16/spb3.0/"

#model_dir = "../runs/TNT_XYY_model1_test_mar6/"
#plot_dir = "../runs/TNT_XYY_model1_test_mar6/"

f_models = [
        #'jrand_autoencoder_m3000.h5',
        #'AEs_CR_may16/jrand_AE_kfold0_mbin13.h5',
#'{j_label}_wp_3tev_test.h5',
#'{j_label}_wp_3tev_smallnet_test.h5',
#'{j_label}_wp_3tev_supervised_test.h5',
#'{j_label}_wp_3tev_supervised_smallnet_test.h5',
#'jrand_wp_3tev_test2.h5',
#'jrand_wp_smallnet_3tev_test2.h5',
#'jrand_wp_3tev_smallnet_test.h5',
#'{j_label}_supervised_Wp.h5',
#'{j_label}_supervised_Wp_cliptau1.h5',
#'{j_label}_supervised_Wp_notau1.h5',
#'{j_label}_supervised_XYY.h5',
#'{j_label}_supervised_notau1_XYY.h5',
#'{j_label}_baseline.h5',
#'{j_label}_batch2k.h5',
#'{j_label}_batch8k.h5',
'jrand_kfold0/',
'jrand_kfold1/',
'jrand_kfold2/',
'jrand_kfold3/',
'jrand_kfold4/',

]


labels = [
        #'AE old',
        #'AE new',
        #'supervised 30k params',
        #'supervised 3k params',
        #'supervised',
        #'supervised clip tau1',
        #'supervised no tau1',
        'kfold 0',
        'kfold 1',
        'kfold 2',
        'kfold 3',
        'kfold 4',
        ]




#model types: 0 CNN (one jet), 1 auto encoder, 2 dense (one jet), 3 CNN (both jets), 4 dense (both jets), 5 is VAE 
model_type = [2,2,2,2,2,2]
#model_type = [2,2,2,2,2,2]
num_models = [4,4,4,4,4,4]
#rand_sort = [False, False, False, False, False, False]
rand_sort = [True]*6

#f_models = ["autoencoder_m3500.h5",  "mar2/dense_sig10_TNT1_s%i.h5", "mar2/cwola_hunting_dense_sig10_s%i.h5"]
#f_models = ["autoencoder_m3500.h5",  "mar15_deta/dense_deta_sig025_TNT1_s%i.h5", "mar15_deta/cwola_hunting_dense_deta_sig025_s%i.h5"]
#labels = ["Autoencoder ", "TNT (S/B = 0.25%)", "CWoLa (S/B = 0.25%)"]
#model_type = [1, 2, 2] 

#f_models = ["autoencoder_m2500.h5",  "mar2/dense_sig10_TNT1_s%i.h5", "mar2/cwola_hunting_dense_sig10_s%i.h5"]
#labels = ["Autoencoder ", "TNT (S/B = 1%)", "CWoLa (S/B = 1%)"]
#model_type = [1, 2, 2] 

colors = ["g", "b", "r", "gray", "purple", "pink", "orange", "m", "skyblue", "yellow"]

n_points = 200.

logy= True

need_images = 1 in model_type

if(need_images):
    options.keys = ["j1_images", "j2_images", "jj_images", "j1_features", "j2_features", "jj_features", 'jet_kinematics']
else:
    options.keys = ["j1_features", "j2_features", "jj_features", 'jet_kinematics']


data, _ = load_dataset_from_options(options)
j1_dense_inputs = data['j1_features']
j2_dense_inputs = data['j2_features']
jj_dense_inputs = data['jj_features']

j1_images = j2_images = jj_images = None
if(need_images):
    j1_images = data['j1_images']
    j2_images = data['j2_images']
    jj_images = data['jj_images']
Y = data['label'].reshape(-1)


j1rand_images = j2rand_images = j1rand_dense_inputs = j2rand_dense_inputs = None
if(any(rand_sort)):
    swapping_idxs = np.random.choice(a=[True,False], size = Y.shape[0])
    if(need_images):
        j1rand_images = copy.deepcopy(j1_images)
        j2rand_images = copy.deepcopy(j2_images)
        j1rand_images[swapping_idxs] = j2_images[swapping_idxs]
        j2rand_images[swapping_idxs] = j1_images[swapping_idxs]

    j1rand_dense_inputs = copy.deepcopy(j1_dense_inputs)
    j2rand_dense_inputs = copy.deepcopy(j2_dense_inputs)
    j1rand_dense_inputs[swapping_idxs] = j2_dense_inputs[swapping_idxs]
    j2rand_dense_inputs[swapping_idxs] = j1_dense_inputs[swapping_idxs]



# reading images
#filter signal

sig_effs = []
bkg_effs = []
aucs = []
sics = []
for idx,f in enumerate(f_models):
    if('sig_idx' in f): 
        f = f.format(sig_idx = sig_idx)
    print(idx, f, model_type[idx])
    if(model_type[idx]  <= 2 or model_type[idx] == 5): #classifier on each jet

        if('notau1' in f):
            non_tau_idx = [0,2,3,4,5,6,7]
            j1_dense_inputs_ = j1_dense_inputs[:,non_tau_idx]
            j2_dense_inputs_ = j2_dense_inputs[:,non_tau_idx]
        elif('cliptau1' in f):
            j1_dense_inputs_ = j1_dense_inputs
            j2_dense_inputs_ = j2_dense_inputs
            print(np.amax(j1_dense_inputs[:,1]), np.mean(j1_dense_inputs[:,1]))
            j1_dense_inputs_[:,1] = np.clip(j1_dense_inputs_[:,1], 0., 0.35)
            j2_dense_inputs_[:,1] = np.clip(j2_dense_inputs_[:,1], 0., 0.35)
        else:
            j1_dense_inputs_ = j1_dense_inputs
            j2_dense_inputs_ = j2_dense_inputs

        if(rand_sort[idx]):
            j1_score, j2_score = get_jet_scores(model_dir, f, model_type[idx], j1rand_images, j2rand_images, j1rand_dense_inputs, j2rand_dense_inputs, num_models = num_models[idx])
        else:
            j1_score, j2_score = get_jet_scores(model_dir, f, model_type[idx], j1_images, j2_images, j1_dense_inputs_, j2_dense_inputs_, num_models = num_models [idx])
        Y = Y.reshape(-1)

        j1_QT = QuantileTransformer(copy = True)
        j1_qs = j1_QT.fit_transform(j1_score.reshape(-1,1)).reshape(-1)
        j2_QT = QuantileTransformer(copy = True)
        j2_qs = j2_QT.fit_transform(j2_score.reshape(-1,1)).reshape(-1)

        #j1_qs = quantile_transform(j1_score.reshape(-1,1)).reshape(-1)
        #j2_qs = quantile_transform(j2_score.reshape(-1,1)).reshape(-1)
        #sig_eff = np.array([(Y[(j1_qs > perc) & (j2_qs > perc) & (Y==1)].shape[0])/(Y[Y==1].shape[0]) for perc in np.arange(0.,1., 1./n_points)])
        #bkg_eff = np.array([(Y[(j1_qs > perc) & (j2_qs > perc) & (Y==0)].shape[0])/(Y[Y==0].shape[0]) for perc in np.arange(0.,1., 1./n_points)])
        jj_scores = combine_scores(j1_qs, j2_qs, options.score_comb)



    else:
        jj_scores = get_jj_scores(model_dir, model_name[idx], model_type[idx], jj_images, jj_dense_inputs)

    bkg_eff, sig_eff, thresholds_cwola = roc_curve(Y, jj_scores)
    bkg_eff = np.clip(bkg_eff, 1e-8, 1.)
    sig_eff = np.clip(sig_eff, 1e-8, 1.)
    sig_effs.append(sig_eff)
    bkg_effs.append(bkg_eff)
    sics.append(sig_eff/np.sqrt(bkg_eff))
    aucs.append(auc(bkg_eff, sig_eff))


            


fs = 18
fs_leg = 14

#roc curve
plt.figure(figsize=fig_size)
for i in range(len(labels)):
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
plt.legend(loc="upper left", fontsize= fs_leg)
plt.savefig(options.output+roc_plot_name)
print("Saving file to %s " % (options.output + roc_plot_name))

#sic curve
eff_min = 1e-3
plt.figure(figsize=fig_size)
for i in range(len(labels)):
    mask_ = bkg_effs[i] > eff_min
    print(labels[i], aucs[i], np.amax(sics[i][mask_]))
    plt.plot(bkg_effs[i][mask_], sics[i][mask_], lw=2, color=colors[i], label=labels[i] + (" (AUC = %.3f)" % aucs[i]))
#plt.plot(fpr_cwola, tpr_cwola, lw=2, color="purple", label="CWOLA = %.3f" %(auc_cwola))
#plt.plot(tnt_bkg_eff, tnt_sig_eff, lw=2, color="r", label="TNT Dense = %.3f"%(auc(tnt_bkg_eff,tnt_sig_eff)))
#plt.plot(sup_bkg_eff, sup_sig_eff, lw=2, color="g", label="Sup. Dense = %.3f"%(auc(sup_bkg_eff,sup_sig_eff)))
#plt.plot(ae_bkg_eff, ae_sig_eff, lw=2, color="b", label="Autoencoders = %.3f"%(auc(ae_bkg_eff,ae_sig_eff)))

#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='random chance')
plt.xlim([eff_min, 1.0])
if(sic_max > 0):
    plt.ylim([0,sic_max])
plt.xscale('log')
plt.xlabel('Background Efficiency', fontsize = fs)
plt.ylabel('Significance Improvement', fontsize = fs)
plt.tick_params(axis='x', labelsize=fs_leg)
plt.tick_params(axis='y', labelsize=fs_leg)
plt.grid(axis = 'y', linestyle='--', linewidth = 0.5)
plt.legend(loc="best", fontsize= fs_leg)
plt.savefig(options.output+sic_plot_name)
print("Saving file to %s " % (options.output + sic_plot_name))

del data


