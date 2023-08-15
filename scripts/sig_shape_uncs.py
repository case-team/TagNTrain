import sys
import os
sys.path.append('..')
from utils.TrainingUtils import *
from utils.OptionUtils import *
from limit_set import *
import numpy as np

def fit_sig(options, outdir, mjj, weights):
    if(not os.path.exists(outdir)): os.system("mkdir " + outdir)

    with h5py.File(outdir + 'shape.h5', "w") as f_sig:
        f_sig.create_dataset('mjj', data = mjj, chunks = True, maxshape = (None))
        f_sig.create_dataset('weights', data = weights, chunks = True, maxshape = (None))


    sig_fit_cmd = "python fit_signalshapes.py -i %s -o %s -M %i --dcb-model --fitRange 0.35 >& %s" % (outdir + 'shape.h5', 
                           outdir, options.mjj_sig,  outdir + "sig_fit_log.txt" )

    fit_cmd_setup = "cd ../fitting; source deactivate; eval `scramv1 runtime -sh`;"  
    fit_cmd_after = "cd -; source deactivate; source activate mlenv0"
    full_sig_fit_cmd = fit_cmd_setup + sig_fit_cmd + "; "  + fit_cmd_after

    subprocess.call(full_sig_fit_cmd,  shell = True, executable = '/bin/bash')


def sig_shape_uncs(options):

    base_path = os.path.abspath(".") + "/"
    if(not os.path.exists(options.output)): os.system("mkdir " + options.output)



    options.keys = ['mjj', 'event_info', 'jet_kinematics', 'sys_weights']
    options.deta_min = 1.3
    options.batch_start = 0
    options.batch_stop = 39
    options.num_total_batches = 40
    if(len(options.sig_file) > 0):
        sig_only_data = load_signal_file(options)
    else:
        print("No sig file")
        exit(1)


    sig_deta = sig_only_data['jet_kinematics'][:,1]
    sig_deta_mask = sig_deta >= options.deta_min
    mjj_nom = sig_only_data['jet_kinematics'][:,0][sig_deta_mask]
    sig_jet_kinematics = sig_only_data['jet_kinematics'][:][sig_deta_mask]
    sys_weights = sig_only_data['sys_weights'][:][sig_deta_mask]
    weights_nom = sys_weights[:,0]

    odir_nom = base_path + options.output + 'sig_shape_nom/'

    fit_sig(options, odir_nom, mjj_nom, weights_nom)
    fname = odir_nom + "sig_fit_%i.json" % options.mjj_sig
    with open(fname, "rb") as f:
        params_nom = json.load(f, encoding = 'latin-1')


    jme_vars = ['JES_up', 'JES_down','JER_up','JER_down']
    mjj_min = 1450.
    mjj_max = 7000.
    variations = []
    print('nom', params_nom['mean'], params_nom['sigma'])

    for jme_var in jme_vars:

        j1_pt_var =   sig_only_data["j1_JME_vars"][:, JME_vars_map["pt_" + jme_var]][sig_deta_mask]


        j2_pt_var =   sig_only_data["j2_JME_vars"][:, JME_vars_map["pt_" + jme_var]][sig_deta_mask]

        sig_jet_kinematics [:, 2] = j1_pt_var
        sig_jet_kinematics [:, 6] = j2_pt_var
        mjj_var = np.clip(mjj_from_4vecs(sig_jet_kinematics[:, 2:6], sig_jet_kinematics[:, 6:10]), mjj_min, mjj_max)


        odir_var = base_path + options.output + jme_var + "/"

        fit_sig(options, odir_var, mjj_var, weights_nom)

        fname = odir_var + "sig_fit_%i.json" % options.mjj_sig

        with open(fname, "rb") as f:
            params_var = json.load(f, encoding = 'latin-1')

        params_var['var'] = jme_var
        print(jme_var, params_var['mean'], params_var['sigma'])

        variations.append(params_var)


    d_means = []
    d_sigmas = []
    for var in variations:
        diff_mean = abs(params_nom['mean'] - var['mean'])/ params_nom['mean']
        diff_sigma = abs(params_nom['sigma'] - var['sigma'])/ params_nom['sigma']
        print(diff_mean, diff_sigma)
        d_means.append(diff_mean)
        d_sigmas.append(diff_sigma)

    mean_unc = np.amax(d_means)
    sigma_unc = np.amax(d_sigmas)
    print("Final Values: Mean uncertainty %.3f, sigma uncertainty %.3f" % (mean_unc, sigma_unc))


if(__name__ == "__main__"):
    parser = input_options()
    options = parser.parse_args()
    sig_shape_uncs(options)
