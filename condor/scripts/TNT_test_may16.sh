#!/bin/bash

set -x

cd CASE/TagNTrain/training
xrdcp root://cmseos.fnal.gov//store/user/oamram/case/h5_files/2017/BB_images_v1.h5 . 
xrdcp root://cmseos.fnal.gov//store/user/oamram/case/models/j1_autoencoder.h5 . 
xrdcp root://cmseos.fnal.gov//store/user/oamram/case/models/j2_autoencoder.h5 . 

#python3 tag_and_train.py -i BB_images_v1.h5 --mjj_low 2250 --mjj_high 2750 -l j2_autoencoder.h5 --mjj_cut --sig_frac 0.0005 --num_data 4000000 -j 1 --iter 0 -o j1_TNT0.h5 --num_epoch 10
#xrdcp -f j1_TNT0.h5 ${1}/j1_TNT0.h5

#python3 tag_and_train.py -i BB_images_v1.h5 --mjj_low 2250 --mjj_high 2750 -l j1_TNT0.h5 --mjj_cut --sig_frac 0.0005 --num_data 4000000 -j 2 --iter 1 -o j2_TNT1.h5 --sig_cut 90 --num_epoch 10
#xrdcp -f j2_TNT1.h5 ${1}/j2_TNT1.h5

python3 tag_and_train.py -i BB_images_v1.h5 --mjj_low 2250 --mjj_high 2750 -l j1_autoencoder.h5 --mjj_cut --sig_frac 0.0005 --num_data 4000000 -j 2 --iter 0 -o j2_TNT0.h5 --num_epoch 10
xrdcp -f j2_TNT0.h5 ${1}/j2_TNT0.h5

python3 tag_and_train.py -i BB_images_v1.h5 --mjj_low 2250 --mjj_high 2750 -l j2_TNT0.h5 --mjj_cut --sig_frac 0.0005 --num_data 4000000 -j 1 --iter 1 -o j1_TNT1.h5 --sig_cut 90 --num_epoch 10
xrdcp -f j1_TNT1.h5 ${1}/j1_TNT1.h5

