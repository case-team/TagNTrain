import pickle
import sys
sys.path.append('..')
from training.train_cwola_hunting_network import *
from training.tag_and_train import *




def train_from_param_dict(fname):

    options = get_options_from_pkl(fname)
    if(options.do_TNT): tag_and_train(options)
    else: train_cwola_hunting_network(options)


if __name__ == "__main__":
    if(len(sys.argv) !=2):
        print("Requires exactly 1 input arg (dict filename)")
        print(sys.argv)
        sys.exit(1)
    train_from_param_dict(sys.argv[1])
