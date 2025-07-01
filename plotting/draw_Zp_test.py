import sys
sys.path.append('..')
from utils.TrainingUtils import *
from optparse import OptionParser
from optparse import OptionGroup

def get_dRs(gen_eta_phi, j_4vec):
    dR = np.sqrt(np.square(gen_eta_phi[:,0] - j_4vec[1]) + 
            np.square(ang_dist(gen_eta_phi[:,1], j_4vec[2] )))
    return dR

def ang_dist(phi1, phi2):
    phi1 = phi1 % (2. * np.pi)
    phi2 = phi2 % (2. * np.pi)
    dphi = phi1 - phi2
    if(len(dphi.shape) > 0):
        dphi[dphi < -np.pi] += 2.*np.pi
        dphi[dphi > np.pi] -= 2.*np.pi
    else:
        if(dphi < -np.pi): dphi += 2.*np.pi
        if(dphi > np.pi): dphi -= 2.*np.pi

    return dphi

parser = input_options()
options = parser.parse_args()

if(not os.path.exists(options.output)): os.system("mkdir %s" % options.output)


options.keys = ['mjj', 'j1_features', 'j2_features', 'jet_kinematics', 'gen_info', 'event_info', 'sys_weights']
do_both_js = True

options.batch_start = 0
options.batch_stop = 40
data = load_signal_file(options)


gen_parts = data['gen_info'][()]
mjj = data['jet_kinematics'][:,0]
j1_4vec = data['jet_kinematics'][:,2:6]
j2_4vec = data['jet_kinematics'][:,2:6]
old_is_lep = data['event_info'][:,4] < 0.5
n_evts = gen_parts.shape[0]
gen_parts_eta_phi = gen_parts[:,:,1:3]
gen_pdg_id = np.abs(gen_parts[:,:,3])
#neutrino pdg ids are 12,14,16
is_lep = gen_pdg_id > 10
#not_neutrinos = ((~np.isclose(gen_pdg_id, 12)) & (~np.isclose(gen_pdg_id, 14)) & (~np.isclose(gen_pdg_id, 16)))
not_neutrinos = (gen_pdg_id < 11)
no_evt_neutrinos = np.all(not_neutrinos, axis = 1)

j1_nprongs = []
j2_nprongs = []
for i in range(mjj.shape[0]):

    j1_dRs = get_dRs(gen_parts_eta_phi[i], j1_4vec[i])
    j2_dRs = get_dRs(gen_parts_eta_phi[i], j2_4vec[i])

    j1_nprongs.append(np.sum(j1_dRs < 0.8))
    j2_nprongs.append(np.sum(j2_dRs < 0.8))


j1_nprongs = np.array(j1_nprongs)
j2_nprongs = np.array(j2_nprongs)

prong_cut = (j1_nprongs > 3) & (j2_nprongs > 3)


print(np.mean(not_neutrinos), np.mean(no_evt_neutrinos), np.mean(old_is_lep), np.mean(prong_cut))

#h_range = (2000, 6000)
h_range = (1500, 4000)


n_bins = 100
normalize = True
make_histogram([mjj, mjj[old_is_lep], mjj[prong_cut], mjj[prong_cut & no_evt_neutrinos]], 
        ["Regular mjj", "Exclude decays with neutrinos", "prongs>=4 cut", "prongs and neutrino cut"], 
        ["red", "blue", "green", "purple" ], xaxis_label = "Mjj", num_bins = n_bins,  
            h_range = h_range, normalize=normalize, fname=options.output + "mjj_filters" + ".png", )

make_histogram([np.abs(gen_pdg_id).reshape(-1)], [""], ["blue"], xaxis_label = "PDG ID", num_bins = 40,  normalize=normalize, fname=options.output + "PDG_ID" + ".png")

make_histogram([j1_nprongs, j2_nprongs], ["J1", "J2"], ["blue", "red"], xaxis_label = "nProngs", num_bins = 9,  normalize=normalize, fname=options.output + "nprongs" + ".png", h_range = (-0.5,8.5))


sig_nosel_fname = options.output + "sig_shape_nolep.h5"
with h5py.File(sig_nosel_fname, "w") as f_sig_shape:
    f_sig_shape.create_dataset('mjj', data = mjj[no_evt_neutrinos], chunks = True, maxshape = (None))
    f_sig_shape.create_dataset('truth_label', data = np.ones((mjj[no_evt_neutrinos].shape[0],1)), chunks = True, maxshape = (None, 1))

