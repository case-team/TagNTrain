#!/bin/bash

set -x

cd CASE/CASEUtils

#xrdcp root://cmsxrootd.fnal.gov//store/user/oamram/case/BBs/BB_UL_MC_v4/BB_batch${2}.h5 temp.h5
xrdcp root://cmsxrootd.fnal.gov//store/user/oamram/case/BBs/DATA/data_batch${2}.h5 temp.h5
#python jet_images/make_jet_images.py -i temp.h5 -o BB_images_batch${2}.h5 
#python jet_images/make_jet_images.py -i temp.h5 -o BB_images_batch${2}.h5  --deta 1.31
#python jet_images/make_jet_images.py -i temp.h5 -o BB_images_batch${2}.h5 --deta 2.5 --deta_min 2.0
python jet_images/make_jet_images.py -i temp.h5 -o data_images_batch${2}.h5 --deta 2.5 --deta_min 2.0

#xrdcp -f BB_images_batch${2}.h5 ${1}
xrdcp -f data_images_batch${2}.h5 ${1}
