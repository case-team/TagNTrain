import sys, commands, os, fnmatch
import math

def print_and_do(s):
    print(s)
    return os.system(s)

samples = [
'QCD_MuEnriched_Pt80to120',
'QCD_MuEnriched_Pt120to170',
'QCD_MuEnriched_Pt170to300',
'QCD_MuEnriched_Pt300to470',
'QCD_MuEnriched_Pt470to600',
'QCD_MuEnriched_Pt600to800',
'QCD_MuEnriched_Pt800to1000',
'QCD_MuEnriched_Pt1000toInf',]

max_files = -1

tag = "_june28"

files_per_batch = 10


eos_base = "root://cmseos.fnal.gov/"
temp_eos_dir = "/store/user/oamram/case/Lund_files/2016APV/Skimmed/"
flist_dir = "lxplus_files_2016APV/"
#outdir = "./"




for sample in samples:

    fl = open(flist_dir + sample + ".txt")
    file_list = fl.read().splitlines()
    fl.close()

    print(sample)
    num_batches = int(math.ceil(float(len(file_list))/files_per_batch))
    print(num_batches)


    if(not os.path.exists("mkdir /eos/uscms" + temp_eos_dir + sample)):
        print_and_do("mkdir /eos/uscms" + temp_eos_dir + sample)

    outdir  = eos_base + temp_eos_dir + sample
    

    for b in range(num_batches):
        label = sample + "_b" + str(b)

        start_file = b*files_per_batch
        end_file = min((b+1)*files_per_batch, len(file_list))
        batch_flist = file_list[start_file : end_file]

        script_name = "scripts/script_temp.sh"
        print_and_do("cp scripts/postprocess_template.sh %s" % script_name)
        fs = open(script_name, "a")

        for f in batch_flist:
            fs.write("xrdcp %s . \n" % f)
        
        cmd = "python postprocess.py -l %s --max_files %i --outdir %s --inputFiles " % (label, max_files, outdir)
        for f in batch_flist: 
            f_local = f.split("/")[-1]
            cmd += f_local + " " 

        fs.write(cmd)
        fs.close()
        print_and_do("chmod +x %s" % script_name)
        print_and_do("python doCondor.py --njobs 1  --overwrite --cmssw -s %s -n Nano_postprocess_%s --mem 6000 --sub"  % (script_name, label + tag))
