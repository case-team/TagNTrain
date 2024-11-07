#!/bin/bash


cd CASE/TagNTrain/
mkdir data
mv ${_CONDOR_SCRATCH_DIR}/models.tar .
tar -xvf models.tar

set -x

python3 scripts/dataset_copier.py -i /store/user/oamram/case/BBs/BB_NAME/ -o data/BB --batchStart BSTART --batchStop BSTOP  --sig_file_out data/ SIGFILE SIG2FILE
mkdir temp
cd temp

source /cvmfs/cms-lpc.opensciencegrid.org/sl7/gpu/Setup.sh
source /cvmfs/cms-lpc.opensciencegrid.org/sl7/gpu/anaconda3/bin/activate mlenv0

python3 ../scripts/selection.py ${_CONDOR_SCRATCH_DIR}/select_opts_KFOLDNUM.json

xrdcp -f FNAME ${1}
xrdcp -f FNAME.npz ${1}
xrdcp -f sig_shape* ${1}
