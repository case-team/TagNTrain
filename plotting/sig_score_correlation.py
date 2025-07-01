import sys
import subprocess
sys.path.append('..')
from utils.TrainingUtils import *
import matplotlib.colors as mcolors


def sig_score_correlation(options):

    options.randsort = True
    options.keys = ['j1_AE_scores', 'j2_AE_scores']
    sig_only_data = load_signal_file(options)

    j1_scores = sig_only_data['j1_AE_scores']
    j2_scores = sig_only_data['j2_AE_scores']

    j1_scores /= np.amax(j1_scores)
    j2_scores /= np.amax(j2_scores)


    alpha = 0.5
    size = 10.0
    color = 'b'
    plt.scatter(j1_scores,j2_scores, alpha = alpha, c = color, s=size)

    lin_corr = np.corrcoef(j1_scores,j2_scores)[0,1]

    text_str = 'Linear Correlation = %.3f' % lin_corr
    plt.annotate(text_str, xy = (0.03, 0.85),  xycoords = 'axes fraction', fontsize=26)

    plt.xlabel("J1 AE score", fontsize=30)
    plt.ylabel("J2 AE score", fontsize=30)
    plt.tick_params(axis='y', labelsize=24)
    plt.tick_params(axis='x', labelsize=24)

    print("saving %s" % options.output)
    plt.savefig(options.output, bbox_inches = "tight")


if(__name__ == "__main__"):
    parser = input_options()
    options = parser.parse_args()
    sig_score_correlation(options)


