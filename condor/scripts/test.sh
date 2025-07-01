#!/bin/bash


cd CASE/TagNTrain/
mkdir data
python3 scripts/dataset_copier.py -i /store/user/oamram/case/BBs/BB_UL_MC_v4 -o data/BB --batchStart 0 --batchStop 1

python3 -c "import tensorflow as tf; print(tf.__version__)"

touch foo.txt

xrdcp -f foo.txt ${1}
