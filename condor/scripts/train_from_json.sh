#!/bin/bash

cd CASE/TagNTrain/
mkdir data
#python scripts/dataset_copier.py -i /store/user/oamram/case/BBs/BB_UL_MC_small_v2_deta_images/ -o data/BB --sig_file_out data/ SIGFILE


set -x
python3 scripts/dataset_copier.py -i /store/user/oamram/case/BBs/BB_NAME/ -o data/BB --sig_file_out data/ SIGFILE SIG2FILE
mkdir temp
cd temp

source /cvmfs/cms-lpc.opensciencegrid.org/sl7/gpu/Setup.sh
source /cvmfs/cms-lpc.opensciencegrid.org/sl7/gpu/anaconda3/bin/activate mlenv0

python3 ../scripts/train_from_param_dict.py ${_CONDOR_SCRATCH_DIR}/train_opts_${2}.json

xrdcp -f model${2}.h5 ${1}
