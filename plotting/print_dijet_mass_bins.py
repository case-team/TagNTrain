import sys
sys.path.append('..')
from utils.TrainingUtils import *
import matplotlib.colors as mcolors
from optparse import OptionParser
from optparse import OptionGroup

parser = input_options()
parser.add_argument("--do_ttbar", default=False, action = 'store_true',  help="Draw ttbar")
options = parser.parse_args()


fin = options.fin
plot_dir = options.output

model_type = options.model_type
if(len(options.label)> 0 and options.label[-1] != '_'):
    options.label += '_'


options.keys = ['jet_kinematics']
data, _ = load_dataset_from_options(options)

Y = data['label'].reshape(-1)
mjj = data['jet_kinematics'][:,0]

print("First mass bins: ")
for i in range(1, len(mass_bins1)):
    mask = (mjj > mass_bins1[i-1] ) & (mjj < mass_bins2[i])
    n = mjj[mask].shape[0]
    print("%.0f-%.0f : %i" % (mass_bins1[i-1] , mass_bins1[i], n))

print("2nd mass bins: ")
for i in range(1, len(mass_bins2)):
    mask = (mjj > mass_bins2[i-1] ) & (mjj < mass_bins2[i])
    n = mjj[mask].shape[0]
    print("%.0f-%.0f : %i" % (mass_bins2[i-1] , mass_bins2[i], n))

