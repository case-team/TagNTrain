#!/bin/bash

set -x

cd CASE/TagNTrain/
mkdir data
python scripts/dataset_copier.py -i /store/user/oamram/case/BBs/BB_UL_MC_small_v2_deta_images/ -o data/BB
source /cvmfs/cms-lpc.opensciencegrid.org/sl7/gpu/Setup.sh
source /cvmfs/cms-lpc.opensciencegrid.org/sl7/gpu/anaconda3/bin/activate mlenv0
mkdir temp
cd temp

python3 ../training/train_cwola_hunting_network.py -i ../data/BB/ --sig_idx 1  --sig_per_batch 75 --mjj_low 2250 --mjj_high 2750  --mjj_sig 2500 --batch_start 0 --batch_stop 10  -o test${2}.h5  --d_eta 1.4 --use_one --num_models 5 -j 2 --seeds ${2} --val_batch_start 10 --val_batch_stop 15

xrdcp -f test${2}.h5 ${1}
