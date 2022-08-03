#!/bin/bash

set -x

cd CASE/TagNTrain/
mkdir data
python scripts/dataset_copier.py -i /store/user/oamram/case/BBs/BB_NAME/ -o data/BB --batchStart BSTART --batchStop BSTOP  --sig_file_out data/ SIGFILE
mv ${_CONDOR_SCRATCH_DIR}/models.tar .
tar -xvf models.tar
source /cvmfs/cms-lpc.opensciencegrid.org/sl7/gpu/Setup.sh
source /cvmfs/cms-lpc.opensciencegrid.org/sl7/gpu/anaconda3/bin/activate mlenv0
mkdir temp
cd temp

python3 ../scripts/classifier_selection.py ${_CONDOR_SCRATCH_DIR}/select_opts_KFOLDNUM.json

xrdcp -f FNAME ${1}
xrdcp -f FNAME.npz ${1}
