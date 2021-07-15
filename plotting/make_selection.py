import sys
sys.path.append('..')
from utils.TrainingUtils import *
import h5py

fin = "../data/BB_v2_3500_images/"
batch_start = 30
#batch_stop = 30
batch_stop = 39
#fin = "../data/BB_v2_2500_images/BB_images_testset.h5"
#batch_start = 0
#batch_stop = -1

num_data = -1
#num_data = 228760

data_start = 0
sig_idx = 1
sig_mass = 3500.

delta_eta_cut = -1

#for rough significance only
m_low = 3150
m_high = 3850


single_file = True
hadronic_only = False

output_dir = "../../CASEUtils/fitting/fit_inputs/mar15/"
model_dir = "../models/BB_v2_M3500/"


#model types: 0 CNN (one jet), 1 auto encoder, 2 dense (one jet), 3 CNN (both jets), 4 dense (both jets), 5 is VAE 
f_models = ["autoencoder_m3500.h5", 
            "mar2/dense_sig10_TNT1_s%i.h5",  "mar2/dense_sig05_TNT1_s%i.h5",   "mar2/dense_sig025_TNT1_s%i.h5",  "mar2/dense_sig01_TNT1_s%i.h5", 
            "mar2/cwola_hunting_dense_sig10_s%i.h5",  "mar2/cwola_hunting_dense_sig05_s%i.h5", "mar2/cwola_hunting_dense_sig025_s%i.h5", "mar2/cwola_hunting_dense_sig01_s%i.h5"
       ]

#f_models = ["autoencoder_m3500.h5", 
#            "mar15_deta/dense_deta_sig10_TNT1_s%i.h5",  "mar15_deta/dense_deta_sig05_TNT1_s%i.h5",   "mar15_deta/dense_deta_sig025_TNT1_s%i.h5",  "mar15_deta/dense_deta_sig01_TNT1_s%i.h5", 
#            "mar15_deta/cwola_hunting_dense_deta_sig10_s%i.h5",  "mar15_deta/cwola_hunting_dense_deta_sig05_s%i.h5", "mar15_deta/cwola_hunting_dense_deta_sig025_s%i.h5", "mar15_deta/cwola_hunting_dense_deta_sig01_s%i.h5"
#model = f_models[0]
model = "mar2/dense_sig025_TNT1_s%i.h5"
sig_frac = 0.0025
model_type = 2
#f_out = "s1_eta_cut_sig025_nosel.h5"
f_out = "s1_sig025_TNT_eff%.1f.h5"
#f_out = "s1_sig05_testset_TNT_eff%.0f.h5"
effs = [30., 10., 2.]
#effs = [100.]








keys = ['mjj', 'event_info']
if(model_type == 0 or model_type == 1):
    keys += ['j1_images', 'j2_images']
if(model_type == 3 ):
    keys.append('jj_images' )
if(model_type == 2):
    keys += ['j1_features', 'j2_features']
if(model_type == 4):
    keys.append('jj_features' )
print(keys)
#keys = ["j1_images", "j2_images", "jj_images", "j1_features", "j2_features", "jj_features", 'mjj']

data = DataReader(fin, keys = keys, signal_idx = sig_idx, sig_frac = sig_frac, start = data_start, stop = data_start + num_data,  batch_start = batch_start, batch_stop = batch_stop,
        hadronic_only = hadronic_only, m_sig = sig_mass, eta_cut = delta_eta_cut )
data.read()
Y = data['label'].reshape(-1)
mjj = data['mjj']
event_num = data['event_info'][:,0]

batch_size = 1024

if(len(effs) ==1 and effs[0] == 100.):
    scores = None

else:
    if('%' in model): 
        model = model % sig_idx
    if(model_type  <= 2 or model_type == 5): #classifier on each jet
        if(model_type <= 2):
            if(len(model) != 2):
                if('/' not in model):
                    j1_fname = model_dir + "j1_" + model
                    j2_fname = model_dir + "j2_" + model
                else:
                    ins_idx = model.rfind('/')+1
                    j1_fname = model_dir + model[:ins_idx] + "j1_" + model[ins_idx:]
                    j2_fname = model_dir + model[:ins_idx] + "j2_" + model[ins_idx:]
                print(j1_fname, j2_fname)
                j1_model = tf.keras.models.load_model(j1_fname)
                j2_model = tf.keras.models.load_model(j2_fname)
            else:
                j1_model = tf.keras.models.load_model(model_dir + model[0])
                j2_model = tf.keras.models.load_model(model_dir + model[1])

            if(model_type == 0):  #CNN
                j1_images = data['j1_images']
                j2_images = data['j2_images']
                j1_score = j1_model.predict(j1_images, batch_size = batch_size)
                j2_score = j2_model.predict(j2_images, batch_size = batch_size)
            elif(model_type == 1): #autoencoder
                j1_images = data['j1_images']
                j2_images = data['j2_images']
                j1_reco_images = j1_model.predict(j1_images, batch_size=batch_size)
                j1_score =  np.mean(np.square(j1_reco_images - j1_images), axis=(1,2))
                j2_reco_images = j2_model.predict(j2_images, batch_size=batch_size)
                j2_score =  np.mean(np.square(j2_reco_images -  j2_images), axis=(1,2))
            elif(model_type == 2): #dense
                j1_dense_inputs = data['j1_features']
                j2_dense_inputs = data['j2_features']
                j1_score = j1_model.predict(j1_dense_inputs, batch_size = batch_size)
                j2_score = j2_model.predict(j2_dense_inputs, batch_size = batch_size)
        elif(model_type == 5): #VAE
            j1_images = data['j1_images']
            j2_images = data['j2_images']
            j1_model = VAE(0, model_dir = model_dir + "j1_" +  f)
            j1_model.load()
            j1_reco_images, j1_z_mean, j1_z_log_var = j1_model.predict_with_latent(j1_images)
            j1_score = compute_loss_of_prediction_mse_kl(j1_images, j1_reco_images, j1_z_mean, j1_z_log_var)[0]
            j2_model = VAE(0, model_dir = model_dir + "j2_" +  f)
            j2_model.load()
            j2_reco_images, j2_z_mean, j2_z_log_var = j2_model.predict_with_latent(j2_images)
            j2_score = compute_loss_of_prediction_mse_kl(j2_images, j2_reco_images, j2_z_mean, j2_z_log_var)[0]


        j1_score = j1_score.reshape(-1)
        j2_score = j2_score.reshape(-1)


    elif(model_type == 3): #CNN both jets
        jj_images = data['jj_images']
        jj_model = tf.keras.models.load_model(model_dir + f)
        scores = jj_model.predict(jj_images, batch_size = batch_size).reshape(-1)

    elif(model_type == 4): #Dense both jets
        jj_dense_inputs = data['jj_features']
        jj_model = tf.keras.models.load_model(model_dir + f)
        scores = jj_model.predict(jj_dense_inputs, batch_size = batch_size).reshape(-1)

        

for eff in effs:
    print("Will select events with efficiency %.3f" % eff)
    percentile_cut = 100. - eff

    if(eff == 100.):
        mask = mjj> 0.
        output_name = output_dir + f_out
    elif(model_type <3):
        mask = make_selection(j1_score, j2_score, percentile_cut)
        output_name = output_dir + (f_out % eff)
    else:
        thresh = np.percentile(scores, percentile_cut)
        mask = scores > thresh
        output_name = output_dir + (f_out % eff)

    mjj_output = mjj[mask]
    is_sig_output = Y[mask]
    event_num_output = event_num[mask]
    print("Selected %i events" % mjj_output.shape[0])

    in_window = (mjj_output > m_low) & (mjj_output < m_high)
    sig_events = is_sig_output > 0.9
    bkg_events = is_sig_output < 0.1
    S = mjj_output[sig_events & in_window].shape[0]
    B = mjj_output[bkg_events & in_window].shape[0]
    print("Mjj window %f to %f " % (m_low, m_high))
    print("S/B %f, sigificance ~ %.1f " % (float(S)/B, S/np.sqrt(B)))
    print("Outputting to %s" % output_name)


    with h5py.File(output_name, "w") as f:
        f.create_dataset("mjj", data=mjj_output, chunks = True)
        f.create_dataset("truth_label", data=is_sig_output)
        f.create_dataset("event_num", data=event_num_output)




del data


