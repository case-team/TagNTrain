#!/bin/bash

set -x

cd CASE/LundReweighting/
eval `scramv1 runtime -sh`
xrdcp -f root://cmseos.fnal.gov//eos/uscms/store/user/oamram/case/sig_files/QstarToQW_M_5000_mW_80_TuneCP2_13TeV-pythia8_TIMBER.h5 . 
python3 CASE/CASE_add_lund_weights.py  --fin QstarToQW_M_5000_mW_80_TuneCP2_13TeV-pythia8_TIMBER.h5 --num_jobs 1 --job_idx ${2} 
cp  QstarToQW_M_5000_mW_80_TuneCP2_13TeV-pythia8_TIMBER.h5 QstarToQW_M_5000_mW_80_TuneCP2_13TeV-pythia8_TIMBER_Lundv2.h5 
xrdcp -f QstarToQW_M_5000_mW_80_TuneCP2_13TeV-pythia8_TIMBER_Lundv2.h5 ${1} 
