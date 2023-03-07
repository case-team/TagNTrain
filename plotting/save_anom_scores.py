import sys
import os
sys.path.append('..')
from utils.TrainingUtils import *
import subprocess
import h5py
import time
import matplotlib.pyplot as plt

def save_anom_scores(options):
    f_sig = h5py.File(options.sig_file)


    if('j_label' in options.labeler_name):
        print('j_label' in options.labeler_name)
        j1_fname = options.labeler_name.format(j_label = "j1")
        j2_fname = options.labeler_name.format(j_label = "j2")
    else:
        j1_fname = j2_fname = options.labeler_name
    print(j1_fname, j2_fname)


    options.keys = ['mjj', 'event_info']
    if(options.model_type == 0 or options.model_type == 1):
        options.keys += ['j1_images', 'j2_images']
    if(options.model_type == 3 ):
        options.keys.append('jj_images' )
    if(options.model_type == 2):
        options.keys += ['j1_features', 'j2_features']
    if(options.model_type == 4):
        options.keys.append('jj_features' )


    compute_mjj_window(options)
    options.keep_mlow = options.mjj_low
    options.keep_mhigh = options.mjj_high
    data, _ = load_dataset_from_options(options)

    j1_feats = data['j1_features']
    j2_feats = data['j2_features']
    batch_size = 1024


    evt_start = -2000
    #evt_start = 0
    j1_m = np.expand_dims(f_sig['jet_kinematics'][evt_start:, 5], axis=-1)
    j2_m = np.expand_dims(f_sig['jet_kinematics'][evt_start:, 9], axis=-1)
    j1_sig_feats = data.process_feats(f_sig['jet1_extraInfo'][evt_start:])
    j2_sig_feats = data.process_feats(f_sig['jet1_extraInfo'][evt_start:])
    j1_sig_feats = np.append(j1_m, j1_sig_feats, axis = 1)
    j2_sig_feats = np.append(j2_m, j2_sig_feats, axis = 1)

    sig_mjj = f_sig['jet_kinematics'][evt_start:,0]
    sig_deta = f_sig['jet_kinematics'][evt_start:,1]

    print(sig_mjj.shape)
    mask = (sig_mjj > 2700. ) & (sig_mjj < 3300.)  & (sig_deta < 1.3)
    j1_sig_feats = j1_sig_feats[mask]
    j2_sig_feats = j2_sig_feats[mask]
    sig_mjj = sig_mjj[mask]
    print(sig_mjj.shape)

    if(options.randsort):
        rng = np.random
        rng.seed(options.BB_seed)
        swapping_idxs = rng.choice(a=[True,False], size = j1_sig_feats.shape[0])
        j1_sig_feats[swapping_idxs], j2_sig_feats[swapping_idxs] = j2_sig_feats[swapping_idxs], j1_sig_feats[swapping_idxs]

    num_kfolds = 4
    jj_sig_score = []

    for i in range(num_kfolds):

        j1_fname_kfold = j1_fname + str(i) + "/"
        j2_fname_kfold = j2_fname + str(i) + "/"
        
        j1_model = ModelEnsemble(location = j1_fname_kfold, num_models = options.num_models)
        j2_model = ModelEnsemble(location = j2_fname_kfold, num_models = options.num_models)


        j1_scores = j1_model.predict(j1_feats, batch_size = batch_size).reshape(-1)
        j2_scores = j2_model.predict(j2_feats, batch_size = batch_size).reshape(-1)

        j1_QT = QuantileTransformer(copy = True)
        j1_qs = j1_QT.fit_transform(j1_scores.reshape(-1,1)).reshape(-1)
        j2_QT = QuantileTransformer(copy = True)
        j2_qs = j2_QT.fit_transform(j2_scores.reshape(-1,1)).reshape(-1)


        j1_sig_scores = j1_model.predict(j1_sig_feats, batch_size = batch_size).reshape(-1)
        j2_sig_scores = j2_model.predict(j2_sig_feats, batch_size = batch_size).reshape(-1)

        j1_sig_qs = j1_QT.transform(j1_sig_scores.reshape(-1,1)).reshape(-1)
        j2_sig_qs = j1_QT.transform(j2_sig_scores.reshape(-1,1)).reshape(-1)
        scores = combine_scores(j1_sig_qs, j2_sig_qs, options.score_comb)
        jj_sig_score.append(scores)

    jj_sig_score = np.array(jj_sig_score)
    print(jj_sig_score.shape)
    jj_sig_score = np.mean(jj_sig_score, axis = 0)
    print(np.mean(jj_sig_score), np.std(jj_sig_score))


    np.savez(options.output, scores = jj_sig_score, mjj = sig_mjj)



if(__name__ == "__main__"):
    parser = input_options()
    options = parser.parse_args()
    save_anom_scores(options)
