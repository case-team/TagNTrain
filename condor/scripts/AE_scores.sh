#!/bin/bash

set -x

cd CASE/TagNTrain/training
source /cvmfs/cms-lpc.opensciencegrid.org/sl7/gpu/Setup.sh
source /cvmfs/cms-lpc.opensciencegrid.org/sl7/gpu/anaconda3/bin/activate mlenv0

#xrdcp root://cmsxrootd.fnal.gov//store/user/oamram/case/BBs/DATA_deta_images/data_images_batch${2}.h5 .
#python3 add_autoencoder_scores.py -i data_images_batch${2}.h5 --labeler ../models/AEs/AEs_data_SR_june9/jrand_AE_kfold0_mbin2.h5
#xrdcp -f data_images_batch${2}.h5 ${1}

xrdcp root://cmsxrootd.fnal.gov//store/user/oamram/case/BBs/BB_UL_MC_v4_deta_images/BB_images_batch${2}.h5 .
python3 add_autoencoder_scores.py -i BB_images_batch${2}.h5 --labeler ../models/AEs/AEs_sep1/jrand_AE_kfold0_mbin2.h5

xrdcp -f BB_images_batch${2}.h5 ${1}
