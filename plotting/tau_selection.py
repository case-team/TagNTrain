
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

Y = data['label'].reshape(-1)
bkg_evts = Y < 0.1
sig_evts = Y > 0.1
mjj = data['mjj']

event_num = data['event_info'][:,0]

j1_msd = data['j1_features'][:,0]
j2_msd = data['j2_features'][:,0]

j1_tau21 = data['j1_features'][:,1]
j2_tau21 = data['j2_features'][:,1]

j1_tau32 = data['j1_features'][:,2]
j2_tau32 = data['j2_features'][:,2]

j1_deepB = data['j1_features'][:,5]
j2_deepB = data['j2_features'][:,5]

j1_LSF = data['j1_features'][:,4]
j2_LSF = data['j2_features'][:,4]

print(data['j1_features'][0])

tau21_cut_thresh = 0.4
tau32_cut_thresh = 0.6
deepB_thresh = 0.5
msd_thresh = 120


deepB_less = np.minimum(j1_deepB, j2_deepB)
deepB_more = np.maximum(j1_deepB, j2_deepB)

msd_mask = (j1_msd > 50 ) & (j2_msd > 50) #signif plot
#msd_mask = (j1_msd > 300 ) & (j2_msd > 300) #XYY
#msd_mask = (j1_msd > 140 ) & (j2_msd > 70) #Wp
#msd_mask = (j1_msd > 300 ) & (j2_msd > 50) #Wkk
deepB_mask = (deepB_more > deepB_thresh) | (deepB_less > deepB_thresh)



mask = (j1_tau32 < 0.65) & ( j2_tau32 < 0.65) & msd_mask #
#mask = (j1_msd > 0)
#mask = (j1_tau21 < 0.4) & ( j2_tau21 < 0.4) & msd_mask #XYY
#mask = (j1_tau32 < 0.6) &  msd_mask & deepB_mask # Wp
#mask = (((j1_msd > 200.) & (j1_tau32 < 0.5)) | (j1_LSF  > 0.8)) & (
#        ((j2_msd > 60.) & (j2_tau21 < 0.5)) | (j2_LSF  > 0.8))

overall_eff = np.mean(mask & bkg_evts)
mjj_window =  (mjj > options.mjj_low ) & (mjj < options.mjj_high) & bkg_evts
mjj_window_eff = mjj[mjj_window & mask].shape[0] / mjj[mjj_window].shape[0]


sig_eff = np.mean(mask & sig_evts) / np.mean(sig_evts)
improvement = sig_eff / mjj_window_eff**(0.5)

print("Overall eff %.3f" % overall_eff)
print("Mjj window eff %.3f" % mjj_window_eff)
print("Sig eff %.3f" % sig_eff)
print("Improvement %.3f" % improvement)

with h5py.File(options.output, "w") as f_sig:
    f_sig.create_dataset('mjj', data = mjj[mask])
    f_sig.create_dataset('truth_label', data = Y[mask])


