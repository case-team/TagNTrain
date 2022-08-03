import sys
import os
import argparse
import subprocess



parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", default='', help="output for analyzer. This will always be the output for job scripts.")
parser.add_argument("-o", "--outdir", default='', help="output for analyzer. This will always be the output for job scripts.")
parser.add_argument("--batchStart", default = -1,  type = int, help = "Starting batch")
parser.add_argument("--batchStop", default = 40,  type = int, help = "Stop batch")
parser.add_argument("--sig_file", default = "", help = "Name of signal file")
parser.add_argument("--sig_file_out", default = "", help = "Where to put signal file")
options = parser.parse_args()

eos_base = "root://cmseos.fnal.gov/"


file_list = subprocess.check_output("xrdfs %s ls " % eos_base + options.input, shell =True).decode("utf-8").split("\n")
os.system("mkdir " + options.outdir)
if(options.batchStart >=0): batch_range = list(range(options.batchStart, options.batchStop))
else: batch_range = list(range(0, len(file_list)))

for f in file_list:
    print(f)
    to_cpy = False
    for b in batch_range:
        if(("batch%i.h5" % b) in f): 
            print(b)
            to_cpy = True

    if(to_cpy):
        os.system("xrdcp %s %s" % (eos_base+f, options.outdir))
    else:
        print("Skipping %s" % f)

if(len(options.sig_file) > 0 and len(options.sig_file_out) > 0):
    sig_file_base = eos_base + "/store/user/oamram/case/sig_files/"
    print("Copying sig file %s %s"  % (sig_file_base + options.sig_file, options.sig_file_out))
    os.system("xrdcp %s %s" % (sig_file_base + options.sig_file, options.sig_file_out))

