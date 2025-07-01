import sys
import os
sys.path.append('..')
from utils.TrainingUtils import *

def copy_taggers(options):

    if(not os.path.exists(options.output)): os.system("mkdir %s" % options.output)

    for k in range (5):

        if(not options.do_TNT):
            in_dir = options.orig_dir + "j1_kfold%i/model*.h5" % k
            out = options.output + "j1_kfold%i/" % k
            if(not os.path.exists(out)): os.system("mkdir %s" % out)
            os.system("cp %s %s" % (in_dir, out))

            in_dir = options.orig_dir + "j2_kfold%i/model*.h5" % k
            out = options.output + "j2_kfold%i/" % k
            if(not os.path.exists(out)): os.system("mkdir %s" % out)
            os.system("cp %s %s" % (in_dir, out))
        else:
            in_dir = options.orig_dir + "jrand_kfold%i/model*.h5" % k
            out = options.output + "jrand_kfold%i/" % k
            if(not os.path.exists(out)): os.system("mkdir %s" % out)
            os.system("cp %s %s" % (in_dir, out))







if(__name__ == "__main__"):
    parser = input_options()
    parser.add_argument("--do_TNT",  default=False, action = 'store_true',  help="Use TNT (default cwola)")
    parser.add_argument("--orig_dir", default = "", help = 'Signal normalization for fit')
    options = parser.parse_args()
    copy_taggers(options)
