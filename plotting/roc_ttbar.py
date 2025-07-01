import sys
sys.path.append('..')
from utils.TrainingUtils import *
import h5py
import time

#fin = "../data/BB_v2_3500_images/BB_images_testset.h5"
#fin = "../data/BB_v2_2500_images/BB_images_testset.h5"
time_start = time.time()

parser = input_options()
options = parser.parse_args([])
options.fin = "../data/BB_UL_MC_v4_deta_images/"
options.batch_start = 40
options.batch_stop = 79
options.sig_idx = 1
options.sig_per_batch = 0
roc_plot_name = "roc.pdf" 
sic_plot_name = "sic.pdf" 
eff_min = 1e-4
#options.keep_mlow = 1400.
#options.keep_mhigh = 2000.
#options.ptsort = True
options.randsort = True

options.hadronic_only = True
options.no_minor_bkgs = False

#options.deta = 1.5
#options.deta = 2.5
#options.deta_min = 2.0


single_file = True

sic_max = 10

plot_dir = "../plots/ttbar_control_region_data_june2/"
model_dir = ""

pt_cut = 400.


f_models = [
        "../plots/ttbar_control_region_mar4/mc_models/supervised.h5",
        #"../plots/ttbar_control_region_mar4/mc_models/model_1st_half.h5",
        #"../plots/ttbar_control_region_mar4/mc_models/model_1st_half_btag.h5",
        "../plots/ttbar_control_region_mar4/data_models/model_1st_half.h5",
        "../plots/ttbar_control_region_mar4/data_models/model_1st_half_btag.h5",
        "../plots/ttbar_control_region_data_june2/145_250_part1/",
        "../plots/ttbar_control_region_data_june2/btag_145_250_part1/",
]

labels = [
         "MC, full supervision",
         #"MC Training: Weak Supervision",
         #"MC Training: Weak Supervision (btag preselection)",
         "$m_{SD}$ 105-220 GeV",
         "$m_{SD}$ 105-220 GeV, b tag",
         "$m_{SD}$ 145-250 GeV",
         "$m_{SD}$ 145-250 GeV, b tag",
         #"Weak Supervision w/ b* inj. (tag: mtop & btag & tau32 < 0.75)",
        ]

heading = "Data, weak supervision"




#model types: 0 CNN (one jet), 1 auto encoder, 2 dense (one jet), 3 CNN (both jets), 4 dense (both jets), 5 is VAE 
model_type = [2,2,2,2,2,2,2,2,2,2]
num_models = [1,1,1,1,1,1,1,1,1,1]
#model_type = [1,1,1,1,1,1]
#num_models = [1,1,1,1,1,1]

#colors = ["g", "b", "r", "gray", "purple", "pink", "orange", "m", "skyblue", "yellow"]
colors = [c_grey, c_purple, c_lightblue, c_red, c_orange]
linestyles=['solid', 'dashed', 'dashdot', 'dotted', (0, (1,1))]

n_points = 200.

logy= True

options.keys = ["j1_features", "j2_features", "jj_features", 'jet_kinematics']


if(single_file):
    data, _ = load_dataset_from_options(options)
    j1_dense_inputs = data['j1_features']
    j2_dense_inputs = data['j2_features']
    jj_dense_inputs = data['jj_features']

    j1_images = j2_images = jj_images = None
    Y = data['label'].reshape(-1)

else:
    print("Deprecated")

# reading images
#filter signal


#ttbar label is -2
sig_events = Y == -2
bkg_events = ~sig_events
Y_ttbar = np.zeros_like(Y)
Y_ttbar[sig_events] = 1



j_label = "j1"

pt_cut_mask = np.maximum(data['jet_kinematics'][:,2], data['jet_kinematics'][:,6]) > 400.

dense_inputs = data[j_label + "_features"][pt_cut_mask]


model_scores = []
for idx,f in enumerate(f_models):

    scores = get_single_jet_scores(model_dir + f_models[idx], model_type[idx], j_images = None, j_dense_inputs = dense_inputs, 
            num_models = num_models[idx], batch_size = 512)
    #hist_scores = [scores[bkg_events], scores[sig_events]]
    #make_histogram(hist_scores, hist_labels, hist_colors, 'Labeler Score', "", 100,
            #normalize = True, save = True, fname = plot_dir + f.replace('.h5', '').replace("/", "") +"_scores.png")
    model_scores.append(scores)
    #eff_cut1 = compute_effcut_metric(scores[sig_region_events], scores[bkg_region_events], eff = 0.01)
    #eff_cut2 = compute_effcut_metric(scores[sig_region_events], scores[bkg_region_events], eff = 0.02)
    #eff_cut3 = compute_effcut_metric(scores[sig_region_events], scores[bkg_region_events], eff = 0.03)
    #print(eff_cut3, eff_cut2, eff_cut1)




make_roc_curve(model_scores, Y_ttbar[pt_cut_mask], labels = labels, colors = colors, logy = True, fname=plot_dir + roc_plot_name)
make_sic_curve(model_scores, Y_ttbar[pt_cut_mask], labels = labels, linestyles=linestyles, heading=heading, heading_pos = 1, colors = colors, ymax = 10.,  eff_min = eff_min, fname=plot_dir + sic_plot_name)
del data
