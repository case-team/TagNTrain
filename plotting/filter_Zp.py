import sys
sys.path.append('..')
from utils.TrainingUtils import *
from optparse import OptionParser
from optparse import OptionGroup


parser = input_options()
options = parser.parse_args()

options.keys = ['mjj', 'j1_features', 'j2_features', 'jet_kinematics', 'gen_info', 'event_info', 'sys_weights']
do_both_js = True

options.batch_start = 0
options.batch_stop = 40
f = h5py.File(options.sig_file, "r")
fout = h5py.File(options.output, "w")

gen_parts = f['gen_info'][()]

gen_pdg_id = np.abs(gen_parts[:,:,3])

not_neutrinos = ((~np.isclose(gen_pdg_id, 12)) & (~np.isclose(gen_pdg_id, 14)) & (~np.isclose(gen_pdg_id, 16)))
no_evt_neutrinos = np.all(not_neutrinos, axis = 1)
mask = no_evt_neutrinos
print(mask.shape)

for key in list(f.keys()):
    if('_eff' in key or f[key].shape[0] == 1):
        out = f[key][()]
    else:
        print(f[key].shape)
        out = f[key][()][mask]

    fout.create_dataset(key, data = out, compression = 'gzip')



fout['preselection_eff'][0] *= np.mean(mask)
for key in list(fout.keys()):
    print(key, fout[key].shape)


f.close()
fout.close()
