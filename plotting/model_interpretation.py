#import shap
import sys
import os
sys.path.append('..')
from utils.TrainingUtils import *
import subprocess
import h5py
import time
import matplotlib.pyplot as plt

def permutation_weights(model, x_eval, x_rand):
    #evaluate difference in model scores when permutting different features

    baseline_vals = model.predict(x_eval).reshape(-1,1)

    x_rep = np.tile(x_eval, (x_rand.shape[0], 1,1))
    x_rand_tile = np.swapaxes(np.tile(x_rand, (x_eval.shape[0], 1,1)),0,1)

    p_weights = np.zeros(x_eval.shape[1])

    #print(x_eval.shape, x_rand.shape, x_rep.shape, x_rand_tile.shape)
    #print("noms", baseline_vals)
    for feat_i in range(x_eval.shape[1]):
        #construct replica's with specific features swaped
        x_rep_c = np.copy(x_rep)
        x_rep_c[:,:,feat_i] = x_rand_tile[:,:,feat_i]

        #evaluate on the replicas
        perm_preds = model.predict(x_rep_c.reshape(-1, x_eval.shape[1])).reshape(x_rand.shape[0], x_eval.shape[0],1)

        #compute difference with respect to nominal prediction
        diffs = perm_preds - baseline_vals
        mags = np.abs(diffs)
        p_weights[feat_i] = np.mean(mags)
    
    return p_weights


def model_interp(options):

    include_bkg_like = False #look at change in scores for most bkg like samples too
    num_top = 100
    num_rand = 100

    if(not os.path.exists(options.output)):
        subprocess.call("mkdir %s" % options.output, shell = True)
    
    if('j_label' in options.labeler_name):
        j1_fname = options.labeler_name.format(j_label = "j1")
        j2_fname = options.labeler_name.format(j_label = "j2")
    else:
        j1_fname = j2_fname = options.labeler_name


    options.keys = ['mjj', 'event_info']
    if(options.model_type == 0 or options.model_type == 1):
        options.keys += ['j1_images', 'j2_images']
    if(options.model_type == 3 ):
        options.keys.append('jj_images' )
    if(options.model_type == 2):
        options.keys += ['j1_features', 'j2_features']
    if(options.model_type == 4):
        options.keys.append('jj_features' )
    #keys = ["j1_images", "j2_images", "jj_images", "j1_features", "j2_features", "jj_features", 'mjj']

    compute_mjj_window(options)
    options.keep_mlow = options.mjj_low
    options.keep_mhigh = options.mjj_high
    options.verbose = False
    data, _ = load_dataset_from_options(options)

    if(options.model_type != 2):
        print("Model type %i not supported" % options.model_type)
        sys.exit(1)


    j1_model = ModelEnsemble(location = j1_fname, num_models = options.num_models)
    j2_model = ModelEnsemble(location = j2_fname, num_models = options.num_models)

    #j1_model_name = j1_fname + "/model0.h5"
    #j2_model_name = j2_fname + "/model0.h5"
    #j1_model = tf.keras.models.load_model(j1_model_name)
    #j2_model = tf.keras.models.load_model(j2_model_name)

    feature_names = [("jet_mass", r" $m_{SD}$ "), 
            ("tau21", r" $\tau_{21}$ "), 
            ("tau32", r" $\tau_{32}$ "), 
            ("tau43", r" $\tau_{43}$ "), 
            ("LSF", "LSF"), 
            ("DeepB", "DeepB Score"), 
            ("nPF", "Num. PF Cands.")]


    Y = data['label']
    j1_feats = data['j1_features']
    j2_feats = data['j2_features']

    mean_j1_feats = np.mean(j1_feats, axis=0)
    mean_j2_feats = np.mean(j1_feats, axis=0)
    std_j1_feats = np.std(j1_feats, axis=0)
    std_j2_feats = np.std(j1_feats, axis=0)

    batch_size = 512
    j1_scores = j1_model.predict(j1_feats, batch_size = batch_size).reshape(-1)
    j2_scores = j2_model.predict(j2_feats, batch_size = batch_size).reshape(-1)

    top_frac = 0.01
    num_top = int(round(top_frac * j1_scores.shape[0]))
    top_j1_scores_idxs = np.argpartition(j1_scores, -num_top)[-num_top:]
    top_j2_scores_idxs = np.argpartition(j2_scores, -num_top)[-num_top:]

    bottom_j1_scores_idxs = np.argpartition(j1_scores, num_top)[:num_top]
    bottom_j2_scores_idxs = np.argpartition(j2_scores, num_top)[:num_top]

    mean_top_j1 = np.mean(j1_feats[top_j1_scores_idxs], axis=0)
    mean_top_j2 = np.mean(j2_feats[top_j2_scores_idxs], axis=0)
    std_top_j1 = np.std(j1_feats[top_j1_scores_idxs], axis=0)
    std_top_j2 = np.std(j2_feats[top_j2_scores_idxs], axis=0)

    def avg_nsubj_ratios(options, feats):
        if(not options.nsubj_ratios):
            eps = 1e-8
            tau21 = feats[:,2] / (feats[:,1] + eps)
            tau32 = feats[:,3] / (feats[:,2] + eps)
            tau43 = feats[:,4] / (feats[:,3] + eps)
        else:
            tau21 = feats[:,2]
            tau32 = feats[:,3]
            tau43 = feats[:,4]

        mean_tau21 = np.mean(tau21)
        mean_tau32 = np.mean(tau32)
        mean_tau43 = np.mean(tau43)

        std_tau21 = np.std(tau21)
        std_tau32 = np.std(tau32)
        std_tau43 = np.std(tau43)
        return ((mean_tau21,mean_tau32,mean_tau43), (std_tau21, std_tau32, std_tau43))

    mean_j1_nsubj, std_j1_nsubj = avg_nsubj_ratios(options, j1_feats)
    mean_top_j1_nsubj, std_top_j1_nsubj = avg_nsubj_ratios(options, j1_feats[top_j1_scores_idxs])
    mean_j2_nsubj, std_j2_nsubj = avg_nsubj_ratios(options, j2_feats)
    mean_top_j2_nsubj, std_top_j2_nsubj = avg_nsubj_ratios(options, j2_feats[top_j2_scores_idxs])

    print("J1: Evaluating based on top %i jets (%.2f signal)"% (num_top, np.mean(Y[top_j1_scores_idxs] > 0)))
    plot_colors = ("b", "r")
    num_bins = 20
    title = ""

    plot_labels = ("Top 1% most anomalous", "All jets")
    for idx,(feat,axis_title) in enumerate(feature_names):
        make_outline_hist( [], ( j1_feats[top_j1_scores_idxs][:,idx], j1_feats[:,idx]), plot_labels, plot_colors, axis_title, title, num_bins, normalize = True, 
                        save = True, fname = options.output + "j1_" + feat + "_topsig_cmp.png")

    print("J2: Evaluating based on top %i jets (%.2f signal)"% (num_top, np.mean(Y[top_j2_scores_idxs] > 0)))
    for idx,(feat,axis_title) in enumerate(feature_names):
        make_outline_hist( [], ( j2_feats[top_j2_scores_idxs][:,idx],  j2_feats[:,idx] ), plot_labels, plot_colors, axis_title, title, num_bins, normalize = True, 
                        save = True, fname = options.output + "j2_" + feat + "_topsig_cmp.png")


    if(include_bkg_like):
        j1_inputs = np.concatenate((j1_feats[top_j1_scores_idxs], j1_feats[bottom_j1_scores_idxs]), axis=0)
        j2_inputs = np.concatenate((j2_feats[top_j2_scores_idxs], j2_feats[bottom_j2_scores_idxs]), axis=0)
    else:
        j1_inputs = j1_feats[top_j1_scores_idxs]
        j2_inputs = j2_feats[top_j2_scores_idxs]

    j1_perm_weights = permutation_weights(j1_model, j1_inputs, j1_feats[:num_rand])
    j2_perm_weights = permutation_weights(j2_model, j2_inputs, j2_feats[:num_rand])
    print("Var: j1 perm weight, j2 perm weight")
    for idx, feat in enumerate(feature_names):
        print("%s: %.3f %.3f" % (feat[0], j1_perm_weights[idx], j2_perm_weights[idx]))
    feat_labels = [feat[1] for feat in feature_names]

    horizontal_bar_chart(j1_perm_weights, feat_labels, fname = options.output + "j1_feature_importance.png", xaxis_label = "Permutation Score")
    horizontal_bar_chart(j2_perm_weights, feat_labels, fname = options.output + "j2_feature_importance.png", xaxis_label = "Permutation Score")





if(__name__ == "__main__"):
    parser = input_options()
    options = parser.parse_args()
    model_interp(options)
