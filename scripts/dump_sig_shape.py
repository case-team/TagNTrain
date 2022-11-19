import sys
sys.path.append('..')
from utils.TrainingUtils import *
import h5py


def dump_sig_shape(options):

    options.keys = ['mjj', 'event_info', 'jet_kinematics', 'sys_weights']
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
    sig_mjj = sig_only_data['jet_kinematics'][:,0][sig_deta_mask]
    sig_weights = sig_only_data['sys_weights'][:,0][sig_deta_mask]

    with h5py.File(options.output, "w") as f_sig:
        f_sig.create_dataset('mjj', data = sig_mjj, chunks = True, maxshape = (None))
        f_sig.create_dataset('weights', data = sig_weights, chunks = True, maxshape = (None))


if(__name__ == "__main__"):
    if(len(sys.argv) ==2): #use a dict of parameters
        fname = sys.argv[1]
        options = get_options_from_json(fname)


    else:
        parser = input_options()
        options = parser.parse_args()

    dump_sig_shape(options)
