import sys, os, fnmatch
import h5py
import numpy as np

def print_and_do(s):
    print(s)
    return os.system(s)

f_inputs = [
        #"XToYYprimeTo4Q_MX3000_MY170_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",

        #"WpToBpT_Wp3000_Bp400_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER.h5", 
        #"XToYYprimeTo4Q_MX3000_MY170_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #"XToYYprimeTo4Q_MX3000_MY170_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5", "XToYYprimeTo4Q_MX3000_MY170_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #"XToYYprimeTo4Q_MX3000_MY25_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5", "XToYYprimeTo4Q_MX3000_MY25_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #"XToYYprimeTo4Q_MX3000_MY25_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5", "XToYYprimeTo4Q_MX3000_MY25_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #"XToYYprimeTo4Q_MX3000_MY400_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5", "XToYYprimeTo4Q_MX3000_MY400_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #"XToYYprimeTo4Q_MX3000_MY400_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #"XToYYprimeTo4Q_MX3000_MY400_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #"WkkToWRadionToWWW_M3000_Mr170_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",                  "XToYYprimeTo4Q_MX3000_MY80_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #"WkkToWRadionToWWW_M3000_Mr400_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",                  "XToYYprimeTo4Q_MX3000_MY80_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #"WkkToWRadionToWWW_M5000_Mr170_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",                  "XToYYprimeTo4Q_MX3000_MY80_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #"WkkToWRadionToWWW_M5000_Mr400_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",                  "XToYYprimeTo4Q_MX3000_MY80_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #             "XToYYprimeTo4Q_MX5000_MY170_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #             "XToYYprimeTo4Q_MX5000_MY170_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #             "XToYYprimeTo4Q_MX5000_MY170_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #             "XToYYprimeTo4Q_MX5000_MY170_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #"WpToBpT_Wp3000_Bp170_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER.h5",             "XToYYprimeTo4Q_MX5000_MY25_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #"WpToBpT_Wp3000_Bp25_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER.h5",              "XToYYprimeTo4Q_MX5000_MY25_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #"XToYYprimeTo4Q_MX5000_MY25_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #"WpToBpT_Wp3000_Bp80_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER.h5",              "XToYYprimeTo4Q_MX5000_MY25_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",



        #"WpToBpT_Wp5000_Bp170_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER.h5",             "XToYYprimeTo4Q_MX5000_MY400_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #"WpToBpT_Wp5000_Bp25_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER.h5",              "XToYYprimeTo4Q_MX5000_MY400_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #"WpToBpT_Wp5000_Bp400_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER.h5",             "XToYYprimeTo4Q_MX5000_MY400_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #"WpToBpT_Wp5000_Bp80_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER.h5",              "XToYYprimeTo4Q_MX5000_MY400_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #  "XToYYprimeTo4Q_MX5000_MY80_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #  "XToYYprimeTo4Q_MX5000_MY80_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #  "XToYYprimeTo4Q_MX5000_MY80_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #  "XToYYprimeTo4Q_MX5000_MY80_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",

        #  "ZpToTpTp_Zp3000_Tp400_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #"ZpToTpTp_Zp5000_Tp400_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
          #"YtoHH_Htott_Y3000_H400_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #"YtoHH_Htott_Y5000_H400_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",

        "QstarToQW_M_3000_mW_170_TuneCP2_13TeV-pythia8_TIMBER.h5",  
        "QstarToQW_M_3000_mW_25_TuneCP2_13TeV-pythia8_TIMBER.h5",   
        "QstarToQW_M_3000_mW_400_TuneCP2_13TeV-pythia8_TIMBER.h5",                                
        "QstarToQW_M_3000_mW_80_TuneCP2_13TeV-pythia8_TIMBER.h5",                                 
        "QstarToQW_M_5000_mW_170_TuneCP2_13TeV-pythia8_TIMBER.h5",                                
        "QstarToQW_M_5000_mW_25_TuneCP2_13TeV-pythia8_TIMBER.h5",                                 
        "QstarToQW_M_5000_mW_400_TuneCP2_13TeV-pythia8_TIMBER.h5",                                
        "QstarToQW_M_5000_mW_80_TuneCP2_13TeV-pythia8_TIMBER.h5",                                 



        #"QstarToQW_M_2000_mW_170_TuneCP2_13TeV-pythia8_TIMBER.h5"
        #"QstarToQW_M_2000_mW_25_TuneCP2_13TeV-pythia8_TIMBER.h5",
        #"QstarToQW_M_2000_mW_400_TuneCP2_13TeV-pythia8_TIMBER.h5"
        #"QstarToQW_M_2000_mW_80_TuneCP2_13TeV-pythia8_TIMBER.h5",


        #"WkkToWRadionToWWW_M2000_Mr170_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",                  
        #"WkkToWRadionToWWW_M2000_Mr400_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",                  
        #"WpToBpT_Wp2000_Bp170_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER.h5",
        #"WpToBpT_Wp2000_Bp25_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER.h5", 
        #"WpToBpT_Wp2000_Bp400_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER.h5",
        #"WpToBpT_Wp2000_Bp80_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER.h5", 



        #"XToYYprimeTo4Q_MX2000_MY400_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5", "XToYYprimeTo4Q_MX2000_MY400_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #"XToYYprimeTo4Q_MX2000_MY80_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5", "XToYYprimeTo4Q_MX2000_MY80_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        #"XToYYprimeTo4Q_MX2000_MY80_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5", "XToYYprimeTo4Q_MX2000_MY80_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",

        # "YtoHH_Htott_Y2000_H400_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        # "ZpToTpTp_Zp2000_Tp400_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        # "XToYYprimeTo4Q_MX2000_MY170_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5", 
        # "XToYYprimeTo4Q_MX2000_MY170_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5", 
        # "XToYYprimeTo4Q_MX2000_MY170_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",
        # "XToYYprimeTo4Q_MX2000_MY170_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5", 
        #"XToYYprimeTo4Q_MX2000_MY25_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5", ,                                
        #"XToYYprimeTo4Q_MX2000_MY25_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",                                   
        #"XToYYprimeTo4Q_MX2000_MY25_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5", ,                                
        #"XToYYprimeTo4Q_MX2000_MY25_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",                                   
        #"XToYYprimeTo4Q_MX2000_MY400_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",                              
        #"XToYYprimeTo4Q_MX2000_MY400_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5",                               

        #"RSGravitonToGluonGluon_kMpl01_M_1000_TuneCP5_13TeV_pythia8_TIMBER.h5",                   
        #"RSGravitonToGluonGluon_kMpl01_M_2000_TuneCP5_13TeV_pythia8_TIMBER.h5",                   
        #"RSGravitonToGluonGluon_kMpl01_M_3000_TuneCP5_13TeV_pythia8_TIMBER.h5",                   
        #"RSGravitonToGluonGluon_kMpl01_M_5000_TuneCP5_13TeV_pythia8_TIMBER.h5",                   


        ]


fdir = "/eos/uscms/store/user/oamram/case/sig_files/"
tag = "_june28"

eos_base = "root://cmseos.fnal.gov/"
eos_local_base = "/eos/uscms/"
#xrootd_base = "root://cmsxrootd.fnal.gov/"
#xrootd_base = "root://cmsxrootd.fnal.gov/"
xrootd_base = "root://xrootd-cms.infn.it//"

mode = 'derive'
#mode = 'get'
#mode = 'transfer'


#odir = "/eos/uscms/store/user/oamram/case/sig_files/LundRW/"
odir = eos_base + "/store/user/oamram/case/sig_files/LundRW/"

memory = 10000
#num_jobs = 15
num_jobs = 1

bad_files = []


for sig_name in f_inputs:
    sig_file = fdir + sig_name


    label = sig_name.split(".")[0]

    if(mode == 'derive'):
        

        cp_sig_cmd = "xrdcp -f %s . \n" % (eos_base + sig_file)
        fout_name = sig_name.replace(".h5", "_Lundv2.h5")
        cmd = "python3 CASE/CASE_add_lund_weights.py  --fin %s --num_jobs %i --job_idx ${2} \n" % (sig_name, num_jobs)
        cp_cmd = "cp  %s %s \n" % (sig_name, fout_name)


        script_name = "scripts/script_temp.sh"

        print_and_do("cp scripts/lund_template.sh %s" % script_name)
        f = open(script_name, "a")
        f.write(cp_sig_cmd)
        f.write(cmd)
        f.write(cp_cmd)

        cp_cmd = "xrdcp -f %s ${1} \n" % (fout_name)
        f.write(cp_cmd)

        f.close()

        print_and_do("chmod +x %s" % script_name)
        print_and_do("python3 doCondor.py --sub --njobs %i --mem %.0f --overwrite -s %s -n LundRW_%s "  % (num_jobs, memory, script_name, label + tag))

    elif(mode == 'get'):
        print("GET")
        oname  = label + "_Lund.h5"
        print("CREATING %s" % oname)

        os.system("cp %s %s" % (sig_file, oname))


        condor_dir = "/eos/uscms/store/user/oamram/Condor_outputs/LundRW_"+ label + tag + "/"

        cmd = "python ../../LundReweighting/CASE_merge_jobs.py %s " % oname
        for i in range(num_jobs):
            cmd += condor_dir + "lund_weights_batch%i.h5 " % i
        os.system(cmd)

        print("CHECKING:")
        f = h5py.File(oname)
        print("orig: ", f['jet_kinematics'][:10,0])
        print("lund: ", f['lund_mjj_check'][:10])
        allclose = np.allclose(f['jet_kinematics'][:,0], f['lund_mjj_check'][:])
        print("allclose: ", allclose)
        if(not allclose):
            bad_files.append(sig_name)

        cmd = "xrdcp -f %s %s/%s" % (oname, odir, oname)
        os.system(cmd)
        os.system("rm %s" % oname)


    elif(mode == 'transfer'):

        fdir = eos_local_base + "/store/user/oamram/case/sig_files/LundRW/"
        oname  = fdir + label + "_Lundv2.h5"
        #lxplus_output =  "/store/group/phys_b2g/CASE/h5_files/UL/merged/Lund/"
        #cmd = "xrdcp %s %s" % (oname, xrootd_base + lxplus_output)
        lxplus_output =  "/eos/cms/store/group/phys_b2g/CASE/h5_files/UL/merged/Lundv2/"
        cmd = "scp %s oamram@lxplus.cern.ch:%s" % (oname, lxplus_output)
        print(cmd)
        os.system(cmd)




print("DONE")
print("BAD FILES")
print(bad_files)
