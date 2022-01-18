import sys
sys.path.append('..')
from utils.TrainingUtils import *
import matplotlib.colors as mcolors
from optparse import OptionParser
from optparse import OptionGroup


parser = input_options()
options = parser.parse_args()

if( len(options.label) == 0):
    print("Must provide plot label (-l, --label)")
    exit(1)

if(options.label[-1] != '_'):
    options.label += '_'

fin = options.fin
plot_dir = options.output
model_dir = options.model_dir

if(plot_dir[-1] != '/'): plot_dir+="/"
if(options.label[-1] != '_'): options.label+= "_"

os.system("mkdir %s" % plot_dir)






use_dense = (options.model_type == 2 or options.model_type == 4)

options.keys = ['jet_kinematics']
if(options.labeler_name != "" and not use_dense):
    options.keys += ['j1_images', 'j2_images']
elif(options.labeler_name != "" and use_dense): 
    options.keys += ["j1_features", "j2_features"]

data, _ = load_dataset_from_options(options)

if(options.labeler_name != ""):
    j1_images = data['j1_images']
    j2_images = data['j2_images']
Y = data['label'].reshape(-1)
mjj = data['jet_kinematics'][:,0]
deta = data['jet_kinematics'][:,1]
j1_m = data['jet_kinematics'][:,5]
j2_m = data['jet_kinematics'][:,9]
j1_images = None
j2_images = None
j1_dense_inputs = None
j2_dense_inputs = None

if(options.labeler_name != "" and not use_dense):
    j1_images = data['j1_images']
    j2_images = data['j2_images']
elif(options.labeler_name != "" and use_dense): 
    j1_dense_inputs = data['j1_features']
    j2_dense_inputs = data['j2_features']


batch_size = 1000





save_figs = True
labels = ['background', 'signal']
#colors = ['b', 'r', 'g', 'purple', 'pink', 'black', 'magenta', 'pink']
colors = ["g", "gray", "b", "r","m", "skyblue", "pink"]
colors_sculpt = []
colormap = cm.viridis
normalize = mcolors.Normalize(vmin = 0., vmax=100.)




sig_events = (Y > 0.9)
bkg_events = (Y < 0.1)

eta_bins = np.linspace(0.,3.0,15)

if(options.labeler_name == ""): #apply selection
    print("Must provide model with --labeler_name option!")
    sys.exit(1)

else:

    if(options.model_type <= 2):
        j1_scores, j2_scores = get_jet_scores("", options.labeler_name, options.model_type, j1_images, j2_images, j1_dense_inputs, j2_dense_inputs, num_models = options.num_models)

        make_profile_hist(deta[bkg_events], j1_scores[bkg_events], eta_bins, xaxis_label = r'$|\Delta\eta|$', yaxis_label = "J1 Score",  
                fname = options.plot_dir + options.label + "j1_eta_dependence.png")
        make_profile_hist(deta[bkg_events], j2_scores[bkg_events], eta_bins, xaxis_label = r'$|\Delta\eta|$', yaxis_label = "J2 Score",  
                fname = options.plot_dir + options.label + "j2_eta_dependence.png")

    else:
        if(options.model_type == 3): #CNN
            jj_model = load_model(model_dir + "jj_" + labeler_name)
            X = np.stack((j1_images_raw,j2_images_raw), axis = -1)
            X = standardize(*zero_center(X))[0]

        if(options.model_type == 4): #Dense
            jj_model = load_model(model_dir + "jj_" + labeler_name)
            X = np.append(j1_dense_inputs, j2_dense_inputs, axis = -1)
            print(X.shape)
            scores = jj_model.predict(X, batch_size = batch_size).reshape(-1)

        jj_scores = jj_model.predict(X, batch_size = batch_size).reshape(-1)


