#!/bin/bash
set -x

cd CASE/TagNTrain
source scripts/condaSetup.sh 
source activate mlenv0

mkdir temp
cd temp
