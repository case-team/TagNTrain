
import sys
sys.path.append('..')
from utils.TrainingUtils import *
from optparse import OptionParser
from optparse import OptionGroup




parser = input_options()
options = parser.parse_args()


compute_mjj_window(options)
options.keep_mlow = options.keep_mhigh = -1
print(options.__dict__)

options.keys = ['mjj', 'event_info', 'j1_features', 'j2_features']

data, _ = load_dataset_from_options(options)
sig_only_data = None
if(len(options.sig_file) > 0):
    sig_only_data = load_signal_file(options)

Y = data['label'].reshape(-1)
bkg_evts = Y < 0.1
mjj = data['mjj']

event_num = data['event_info'][:,0]

j1_msd = data['j1_features'][:,0]
j2_msd = data['j2_features'][:,0]

j1_tau21 = data['j1_features'][:,1]
j2_tau21 = data['j2_features'][:,1]

j1_tau32 = data['j1_features'][:,2]
j2_tau32 = data['j2_features'][:,2]

print(data['j1_features'][0])

tau21_cut_thresh = 0.4
tau32_cut_thresh = 0.7
msd_thresh = 125

msd_mask = (j1_msd > msd_thresh ) & (j2_msd > msd_thresh)

mask = (j1_tau21 < tau21_cut_thresh) & ( j2_tau21 < tau21_cut_thresh) & msd_mask
#mask = (j1_tau32 < tau32_cut_thresh) & ( j2_tau32 < tau32_cut_thresh) & msd_mask

overall_eff = np.mean(mask & bkg_evts)
mjj_window =  (mjj > options.mjj_low ) & (mjj < options.mjj_high) & bkg_evts
mjj_window_eff = mjj[mjj_window & mask].shape[0] / mjj[mjj_window].shape[0]

print("Overall eff %.3f" % overall_eff)
print("Mjj window eff %.3f" % mjj_window_eff)

with h5py.File(options.output, "w") as f_sig:
    f_sig.create_dataset('mjj', data = mjj[mask])
    f_sig.create_dataset('truth_label', data = Y[mask])


