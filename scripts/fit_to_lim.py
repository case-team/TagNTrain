
from limit_set import *


def get_lims(fit_res, preselection_eff, tagging_eff, lumi = 138):
    out = dict()

    out['obs'] = fit_res['obs_lim_events']/lumi/preselection_eff/tagging_eff
    out['exp'] = fit_res['exp_lim_events']/lumi/preselection_eff/tagging_eff
    out['exp+1'] = fit_res['exp_lim_1sig_high']/lumi/preselection_eff/tagging_eff
    out['exp-1'] = fit_res['exp_lim_1sig_low']/lumi/preselection_eff/tagging_eff
    return out

def get_presel_eff(sig_file):
    with h5py.File(sig_file, "r") as f:
        presel_eff = f['preselection_eff'][0] * f['d_eta_eff'][0]
    return presel_eff


def fit_to_lim(options):

    presel_eff = get_presel_eff(options.sig_file)

    with open(options.fit_loc, "rb") as f:
        fit_res = json.load(f, encoding = 'latin-1')

    lims = get_lims(fit_res, presel_eff, options.tag_eff)

    sig_name = options.sig_file.split("/")[-1]
    print(sig_name)

    outfile = options.output + sig_name.replace("_Lund.h5", ".json")

    print(lims)
    print("Writing out to %s" % outfile)
    write_params(outfile, lims)

if(__name__ == "__main__"):
    parser = input_options()
    parser.add_argument("--fit_loc", default = "")
    parser.add_argument("--tag_eff", default = 1.0, type = float)
    options = parser.parse_args()
    fit_to_lim(options)
