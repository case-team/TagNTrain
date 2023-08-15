import sys
sys.path.append('..')
from utils.TrainingUtils import *

from optparse import OptionParser
from optparse import OptionGroup




parser = input_options()
options = parser.parse_args()

if(options.output[-1] != '/'): options.output +='/'

os.system("mkdir %s" %options.output)





#################################################################




f = h5py.File(options.sig_file, 'r')
mjj = f['jet_kinematics'][:,0]

smear_vals = [5,10,20,30]
all_mjjs = [mjj]
n_bins = 30


for val in smear_vals:
    frac = val / 100.
    noise = np.random.randn(*mjj.shape)

    new_mjjs = mjj * (1. + frac * noise)

    new_fname = options.output + options.sig_file.split("/")[-1].replace(".h5", "_smear%i.h5" % val)
    os.system("cp %s %s" % (options.sig_file, new_fname))

    all_mjjs.append(new_mjjs)

    new_f = h5py.File(new_fname, 'r+')
    new_f['jet_kinematics'][:,0] = new_mjjs
    new_f.close()



labels = ['nominal'] + ["Smear %i%%" %val for val in smear_vals]
colors= ['black', 'blue', 'green', 'red', 'orange']
make_histogram(all_mjjs, labels, colors, 'Mjj', num_bins = n_bins,  logy = True,
        h_range = (options.mjj_sig * 0.5, options.mjj_sig * 1.5),
        normalize=True, fname=options.output + 'mjj_dists' + ".png")


f.close()



