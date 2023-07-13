import sys
import os
sys.path.append('..')
from utils.TrainingUtils import *
from utils.OptionUtils import *
from limit_set import *


def inclusive_fit(options):

    base_path = os.path.abspath(".") + "/"

    spbs = copy.copy(options.spbs)
    sigmas = []
    pvals = []
    asimov_sigmas = []
    asimov_pvals = []

    out = dict()
    if(os.path.exists(options.output + "saved_params.json")):
        with open(options.output + "saved_params.json", 'r') as f:
           out = json.load(f, encoding="latin-1")

    else:
        out = dict()
        out['spbs'] = []
        out['signif'] = []
        out['pvals'] = []
        out['asimov_signif'] = []
        out['asimov_pvals'] = []
        out['injected_xsecs'] = []

    for spb in options.spbs:

        d_opts = copy.deepcopy(options)
        d_opts.sig_per_batch = -1
        d_opts.output = options.output + "spb" + str(spb) + "/"
        os.system("mkdir -p "  + d_opts.output)
        d_opts.keys = ['jet_kinematics', 'mjj']
        d_opts.sig_per_batch = spb
        d_opts.mbin = d_opts.keep_mlow  = d_opts.keep_mhigh = -1
        d_opts.verbose = False

        #data,_ = load_dataset_from_options(d_opts)


        #mjj = data["mjj"]
        #label = data["label"]


        #f_tmp = d_opts.output + ("fit_inputs_spb%s.h5" % str(spb))


        #with h5py.File(f_tmp, "w") as f:
        #    f.create_dataset("mjj", data=mjj)
        #    f.create_dataset("truth_label", data=label)


        #run_dijetfit(d_opts, fit_start = -1, sig_shape_file = os.path.abspath(options.sig_shape), input_file = os.path.abspath(f_tmp), output_dir = "", loop = True)
        
        fit_res = get_options_from_json(d_opts.output + "fit_results_%.1f.json" % options.mjj_sig)
        print("\n\n SPB %i " % spb)
        print(fit_res.__dict__)
        sigmas.append(fit_res.signif)
        pvals.append(fit_res.pval)
        asimov_sigmas.append(fit_res.asimov_signif)
        asimov_pvals.append(fit_res.asimov_pval)


    lumi = 26.81
    numBatches = 40
    preselection_eff = 0.
    injected_xsecs = [0]
    if(len(options.sig_file ) > 0):
        with h5py.File(options.sig_file, "r") as f:
            presel_eff = f['preselection_eff'][0]
            deta_eff = f['d_eta_eff'][0]

        preselection_eff = get_preselection_params(options.sig_file)[0]
        injected_xsecs = [(  spb * numBatches / lumi / preselection_eff ) for spb in options.spbs]
    out['spbs'] += spbs
    out['signif'] += sigmas
    out['pvals'] += pvals
    out['asimov_signif'] += asimov_sigmas
    out['asimov_pvals'] += asimov_pvals
    out['injected_xsecs'] += injected_xsecs

    write_params(options.output + "saved_params.json", out)
    

        






if(__name__ == "__main__"):
    parser = input_options()
    parser.add_argument("--spbs", nargs="+", default = [], type = float)
    parser.add_argument("--sig_shape", default = "", type = str)
    options = parser.parse_args()
    inclusive_fit(options)
