#!/bin/bash

set -x

cd PhysicsTools/NanoAODTools/
eval `scramv1 runtime -sh`

#python postprocess.py EOSDIR LABEL
#xrdcp -f hadd_LABEL.root ${1}
