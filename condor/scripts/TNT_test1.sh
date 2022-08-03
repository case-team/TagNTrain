#!/bin/bash

set -x

cd CASE/TagNTrain/training
xrdcp root://cmsxrootd.fnal.gov//store/group/phys_b2g/CASE/h5_files/2017/BB_files_images/BB_test_Wprime.h5 .
python3 train_auto_encoder.py -j 1 -o j1_autoencoder.h5 -i BB_test_Wprime.h5 --num_epoch 15
#python3 train_auto_encoder.py -j 2 -o j2_autoencoder.h5 -i BB_test_Wprime.h5 --num_epoch 15
#python3 tag_and_train.py -j 1 -l  j2_autoencoder.h5  -o j1_TNT0.h5 -i BB_test_Wprime.h5 --iter 0 --num_epoch 20
python3 tag_and_train.py -j 2 -l  j1_autoencoder.h5  -o j2_TNT0.h5 -i BB_test_Wprime.h5 --iter 0  --num_epoch 20


xrdcp -f j1_autoencoder.h5 ${1}/j1_autoencoder.h5
#xrdcp -f j2_autoencoder.h5 ${1}/j2_autoencoder.h5
#xrdcp -f j1_TNT0.h5 ${1}/j1_TNT0.h5
xrdcp -f j2_TNT0.h5 ${1}/j2_TNT0.h5
