import sys
import os
import argparse
import subprocess



parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", default='', help="output for analyzer. This will always be the output for job scripts.")
parser.add_argument("-o", "--outdir", default='', help="output for analyzer. This will always be the output for job scripts.")
parser.add_argument("-n", "--numBatch", default = -1,  type = int, help = "Max number of batches")
options = parser.parse_args()

eos_base = "root://cmseos.fnal.gov/"

file_list = subprocess.check_output("xrdfs %s ls " % eos_base + options.input, shell =True).decode("utf-8")
os.system("mkdir " + options.outdir)
for f in file_list.split("\n"):
    os.system("xrdcp %s %s" % (eos_base+f, options.outdir))

