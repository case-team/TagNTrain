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
options.hadronic_only = False
options.deta = 1.3


sic_max = 10

model_dir = ""

if(not os.path.exists(options.output)): os.system("mkdir %s" % options.output)


f_models = [options.labeler_name]

#f_models = [
#'XYY_5TeV_{j_label}.h5'
#
#]


#labels = [ 'Supervised', ]
labels = [ 'AE', ]




#model types: 0 CNN (one jet), 1 auto encoder, 2 dense (one jet), 3 CNN (both jets), 4 dense (both jets), 5 is VAE 
#model_type = [2,2,2,2,2,2]
#model_type = [4,4,4,4]
#rand_sort = [False, False, False, False, False, False]
model_type = [1]
rand_sort = [True]
num_models = [1]
quantile = [True]

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
    options.keys = ["j1_images", "j2_images", "j1_features", "j2_features", "jj_features", 'jet_kinematics']
else:
    options.keys = ["j1_features", "j2_features", "jj_features", 'jet_kinematics']


data, _ = load_dataset_from_options(options)

sig_only_data = load_signal_file(options)

j1_dense_inputs = data['j1_features']
j2_dense_inputs = data['j2_features']
jj_dense_inputs = data['jj_features']

j1_sig_inputs = sig_only_data['j1_features']
j2_sig_inputs = sig_only_data['j2_features']
jj_sig_inputs = sig_only_data['jj_features']

sig_weights = sig_only_data['sys_weights'][:,0]
sig_weights *= sig_only_data['lund_weights'][:]

nbkg = j1_dense_inputs.shape[0]
nsig = j1_sig_inputs.shape[0]

j1_dense_inputs = np.concatenate([j1_sig_inputs, j1_dense_inputs])
j2_dense_inputs = np.concatenate([j2_sig_inputs, j2_dense_inputs])
jj_dense_inputs = np.concatenate([jj_sig_inputs, jj_dense_inputs])

j1_images = j2_images = jj_images = None
if(need_images):
    j1_images = data['j1_images']
    j2_images = data['j2_images']

    j1_sig_images = sig_only_data['j1_images']
    j2_sig_images = sig_only_data['j2_images']

    j1_images = np.concatenate([j1_sig_images, j1_images], axis = 0)
    j2_images = np.concatenate([j2_sig_images, j2_images], axis = 0)

Y = np.concatenate([np.ones(nsig), np.zeros(nbkg)]).reshape(-1,1)
weights = np.concatenate([sig_weights, np.ones(nbkg)])


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
bces = []

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

        if(quantile[idx]):
            class_weights = np.ones_like(Y)
            sigs = Y > 0.5
            bkgs = ~sigs
            sig_weight = np.sum(bkgs) /np.sum(sigs)
            class_weights[sigs] = sig_weight
            bce1 = log_loss(Y, j1_score, sample_weight = class_weights)
            bce2 = log_loss(Y, j2_score, sample_weight = class_weights)
            print(bce1, bce2)
            bce = (bce1 + bce2) / 2.0
            #bce = log_loss(Y, j1_score * j2_score, sample_weight = class_weights)


            bces.append(bce)

            j1_QT = QuantileTransformer(copy = True)
            j1_qs = j1_QT.fit_transform(j1_score.reshape(-1,1)).reshape(-1)
            j2_QT = QuantileTransformer(copy = True)
            j2_qs = j2_QT.fit_transform(j2_score.reshape(-1,1)).reshape(-1)
        else:
            j1_qs = j1_score
            j2_qs = j2_score


        #j1_qs = quantile_transform(j1_score.reshape(-1,1)).reshape(-1)
        #j2_qs = quantile_transform(j2_score.reshape(-1,1)).reshape(-1)
        #sig_eff = np.array([(Y[(j1_qs > perc) & (j2_qs > perc) & (Y==1)].shape[0])/(Y[Y==1].shape[0]) for perc in np.arange(0.,1., 1./n_points)])
        #bkg_eff = np.array([(Y[(j1_qs > perc) & (j2_qs > perc) & (Y==0)].shape[0])/(Y[Y==0].shape[0]) for perc in np.arange(0.,1., 1./n_points)])
        jj_scores = combine_scores(j1_qs, j2_qs, options.score_comb)



    else:
        jj_scores = get_jj_scores(model_dir, f, model_type[idx], jj_images, jj_dense_inputs)
        bce = log_loss(Y, jj_scores)
        bces.append(bce)

    print(Y.shape, jj_scores.shape, weights.shape)
    bkg_eff, sig_eff, thresholds_cwola = roc_curve(Y, jj_scores, sample_weight = weights)
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
    sic_10 = np.interp(0.1, bkg_effs[idx], sics[idx])
    sic_2p5 = np.interp(0.025, bkg_effs[idx], sics[idx])
    sic_01 = np.interp(0.01, bkg_effs[idx], sics[idx])
    print("SICs at 0.1, 0.025, and 0.01 bkg eff: %.2f %.2f %.2f" % (sic_10, sic_2p5, sic_01))
    print("AUC %.3f" % aucs[i])
    print("BCE %.3f" % bces[i])

    if(i==0):
        out = dict()
        sig_name = options.sig_file.replace("../data/LundRW/", "").replace("_Lundv2", "")
        print(sig_name)
        out[sig_name] = out2 = dict()
        out2['AUC'] = aucs[i]
        out2['BCE'] = bces[i]
        out2['SIC-1%'] = sic_01
        out2['SIC-2.5%'] = sic_2p5
        out2['SIC-10%'] = sic_10
        write_params(options.output + "metrics.json", out)


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


