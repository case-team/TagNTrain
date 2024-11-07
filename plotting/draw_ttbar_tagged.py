import sys
import subprocess
sys.path.append('..')
from utils.TrainingUtils import *
import matplotlib.colors as mcolors
from optparse import OptionParser
from optparse import OptionGroup

def draw_ttbar_tagged(options):

    if(not os.path.exists(options.output)):
        subprocess.call("mkdir %s" % options.output, shell = True)

    fin = options.fin
    plot_dir = options.plot_dir
    model_dir = options.model_dir

        

    options.keys = ['j1_features', 'j2_features', 'jet_kinematics']
    if(len(options.sig_file) == 0): 
        options.sig_per_batch = 0
    #options.keep_mlow = 1300.
    #options.keep_mhigh = 1800.
    #options.keep_mhigh = 99999.
    if(not options.randsort): options.ptsort = True


    data, _ = load_dataset_from_options(options)

    Y = data['label'].reshape(-1)
    mjj = data['jet_kinematics'][:,0]
    j1_pt = data['jet_kinematics'][:,2]
    j2_pt = data['jet_kinematics'][:,6]

    j1_m = data['j1_features'][:,0]
    j2_m = data['j2_features'][:,0]

    j1_tau21 = data['j1_features'][:,2]
    j1_tau32 = data['j1_features'][:,3]
    j1_deepcsv = data['j1_features'][:,-2]

    j2_tau21 = data['j2_features'][:,2]
    j2_tau32 = data['j2_features'][:,3]
    j2_deepcsv = data['j2_features'][:,-2]

    #QCD: 0 Single Top: -1, ttbar: -2, V+jets: -3

    sig = (Y == 1)
    QCD = (Y == 0)
    ttbar = (Y == -2)
    other = (Y == -1) | (Y == -3)

    pt_cut = (j1_pt > 400.) & (j2_pt > 400.)

    #proper cuts are UL16preVFP 0.2027, UL16PostVFP 0.1918, UL17 0.1355, UL18 0.1208
    #https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
    # use an average

    #lower pt side is used to 'tag'
    pre_tag_selection = pt_cut & (j2_tau32 < options.tau32_cut) & (j2_deepcsv > options.deepcsv_cut)


    probe_scores = get_single_jet_scores(options.labeler_name, options.model_type, j_images = None, j_dense_inputs = data['j1_features'][pre_tag_selection], 
            num_models = options.num_models, batch_size = 512)


    eff = 1.0
    percentile_cut = 100. - eff
    thresh = np.percentile(probe_scores, percentile_cut)

    nn_cut = np.copy(pre_tag_selection)
    nn_cut[pre_tag_selection] = probe_scores > thresh



    pass_cut = pre_tag_selection & nn_cut
    fail_cut = pre_tag_selection & ~nn_cut

    print(np.mean(pre_tag_selection))
    print(np.mean(pass_cut))
    #print(np.mean(probe_scores > thresh))


    ttbar_frac_before = j2_m[pre_tag_selection & ttbar].shape[0] / j2_m[pre_tag_selection].shape[0]
    ttbar_frac_after = j2_m[pass_cut & ttbar].shape[0] / j2_m[pass_cut].shape[0]


    sig_frac_before = j2_m[pre_tag_selection & sig].shape[0] / j2_m[pre_tag_selection].shape[0]
    sig_frac_after = j2_m[pass_cut & sig].shape[0] / j2_m[pass_cut].shape[0]

    print("Before: %i events, %.3f ttbar frac %.3f sig frac" % (j2_m[pre_tag_selection].shape[0], ttbar_frac_before, sig_frac_before))
    print("After tag: %i events, %.3f ttbar frac %.3f sig frac" % (j2_m[pass_cut].shape[0], ttbar_frac_after, sig_frac_before))


    save_figs = True
    labels = ['QCD', 'ttbar', 'tW and V+jets']
    colors = ["b", "r", "g"] # "m", "skyblue", "pink"]

    masks = [QCD, ttbar, other]

    if(sig_frac_before > 0):
        labels.append("Signal")
        colors.append("m")
        masks.append(sig)



    n_m_bins = 28
    m_range = (5, 285)
    n_feat_bins = 10
    feat_range = (0., 1.)

    def hist_list(var, cut, masks):
        return [var[cut & mask] for mask in masks]

    make_histogram(hist_list(j1_m, pre_tag_selection, masks), labels, colors, logy=True, xaxis_label = 'J1 Jet Mass', title = "Pre-Selection", 
        num_bins = n_m_bins, h_range = m_range, stacked = True, fname = options.output + "jet_mass_presel_tag.png")

    make_histogram(j2_m[pass_cut & ttbar], ['ttbar'], ['r'], xaxis_label = 'J2 Jet Mass', title = "Selected ttbar", 
        num_bins = n_m_bins, h_range = m_range, stacked = True, fname = options.output + "jet_mass_pass_ttbar.png")

    make_histogram(j2_m[pre_tag_selection & ttbar], ['ttbar'], ['r'], xaxis_label = 'J2 Jet Mass', title = "Pre-sel ttbar", 
        num_bins = n_m_bins, h_range = m_range, stacked = True, fname = options.output + "jet_mass_presel_ttbar.png")

    make_histogram(j2_m[pass_cut & other], ['other'], ['g'], xaxis_label = 'J2 Jet Mass', title = "Selected Z+jets and other", 
        num_bins = n_m_bins*2, h_range = m_range, stacked = True, fname = options.output + "jet_mass_pass_z.png")

    make_histogram(hist_list(j2_m, pass_cut, masks), labels, colors, xaxis_label = 'J2 Jet Mass', title = " Pass J1 Top-Tag", 
        num_bins = n_m_bins, h_range = m_range, stacked = True, fname = options.output + "jet_mass_pass_probe.png")


    make_histogram(hist_list(j2_m, pass_cut, masks), labels, colors, logy=True, xaxis_label = 'J2 Jet Mass', title = " Pass J1 Top-Tag", 
        num_bins = n_m_bins, h_range = m_range, stacked = True, fname = options.output + "jet_mass_pass_probe.png")

    make_histogram(hist_list(j2_m, fail_cut, masks), labels, colors, logy=True, xaxis_label = 'J2 Jet Mass', title = " 'Fail J1 Top-Tag' ", 
        num_bins = n_m_bins, h_range = m_range, stacked = True, fname = options.output + "jet_mass_fail_probe.png")

    make_histogram(hist_list(j1_m, pass_cut, masks), labels, colors, logy=True, xaxis_label = 'J1 Jet Mass', title = "Pass J1 Top-Tag", 
        num_bins = n_m_bins, h_range = m_range, stacked = True, fname = options.output + "jet_mass_pass_tag.png")



    f_tagged_out = options.output + "tagged_masses.h5"
    f_raw_out = options.output + "masses.h5"
    f_top_out = options.output + "top_only_masses.h5"
    print("Creating %s %s %s" % (f_raw_out, f_tagged_out, f_top_out))

    with h5py.File(f_tagged_out, "w") as f:
        f.create_dataset('mass_pass', data=j2_m[pass_cut])
        f.create_dataset('mass_fail', data=j2_m[fail_cut])
        f.create_dataset('label', data=Y[pass_cut])

    with h5py.File(f_raw_out, "w") as f:
        f.create_dataset('mass', data=j2_m[pre_tag_selection])
        f.create_dataset('label', data=Y[pre_tag_selection])

    with h5py.File(f_top_out, "w") as f:
        f.create_dataset('mass', data=j2_m[pre_tag_selection & ttbar])

if(__name__ == "__main__"):
    parser = input_options()
    parser.add_argument("--tau32_cut", default = 999., type = float, help = "What tau32 cut to use on tag side")
    parser.add_argument("--deepcsv_cut", default = -999., type = float, help = "What deepcsv cut to use on tag side")
    options = parser.parse_args()
    draw_ttbar_tagged(options)
