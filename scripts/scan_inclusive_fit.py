
import sys
import os
sys.path.append('..')
from utils.TrainingUtils import *
from utils.OptionUtils import *
from full_scan import *


def scan_inclusive_fit(options):

    sig_masses = list(range(1800,5800, 100))


    if(options.step == 'fit'):
        for sig_mass in sig_masses:
            d_opts = copy.deepcopy(options)
            d_opts.mbin = d_opts.keep_mlow  = d_opts.keep_mhigh = -1
            d_opts.verbose = False
            d_opts.mjj_sig = sig_mass

            base_path = os.path.abspath(".") + "/"
            sig_shape_file = base_path + "../fitting/interpolated_signal_shapes/case_interpolation_M%.1f.root" % sig_mass

            run_dijetfit(d_opts, fit_start = -1, sig_shape_file = sig_shape_file, input_file = os.path.abspath(options.fin), output_dir = "", loop = True)

    elif(options.step == 'plot'):
        fit_files = [options.output + 'fit_results_%i.0.json' % sig_mass for sig_mass in sig_masses]
        plot_significances(fit_files, options.output, sig_masses = sig_masses, SR_lines = False)



if(__name__ == "__main__"):
    parser = input_options()
    parser.add_argument("--step", default = "fit",  help = 'Which step to perform (train, get, select, fit, roc, all)')
    options = parser.parse_args()
    scan_inclusive_fit(options)
