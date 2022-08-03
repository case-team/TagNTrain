
#!/bin/bash

set -x

cd CASE/TagNTrain/training
xrdcp root://cmsxrootd.fnal.gov//store/group/phys_b2g/CASE/h5_files/2017/QCD_only.h5 .
python3 train_auto_encoder.py -j 1 -o j1_autoencoder.h5 -i QCD_only.h5 --num_epoch 20 --num_data -1 


xrdcp -f j1_autoencoder.h5 ${1}/j1_autoencoder.h5
