import sys, commands, os, fnmatch
import h5py
import numpy as np

def print_and_do(s):
    print(s)
    return os.system(s)

f_inputs = [
        "XToYYprimeTo4Q_MX3000_MY170_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",

        "WpToBpT_Wp3000_Bp400_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lund.h5", 
        "XToYYprimeTo4Q_MX2000_MY400_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5", "XToYYprimeTo4Q_MX2000_MY400_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "XToYYprimeTo4Q_MX2000_MY80_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5", "XToYYprimeTo4Q_MX2000_MY80_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "XToYYprimeTo4Q_MX2000_MY80_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5", "XToYYprimeTo4Q_MX2000_MY80_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "XToYYprimeTo4Q_MX3000_MY170_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "XToYYprimeTo4Q_MX3000_MY170_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5", "XToYYprimeTo4Q_MX3000_MY170_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "XToYYprimeTo4Q_MX3000_MY25_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5", "XToYYprimeTo4Q_MX3000_MY25_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "XToYYprimeTo4Q_MX3000_MY25_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5", "XToYYprimeTo4Q_MX3000_MY25_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "XToYYprimeTo4Q_MX3000_MY400_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5", "XToYYprimeTo4Q_MX3000_MY400_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "WkkToWRadionToWWW_M2000_Mr170_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",                  "XToYYprimeTo4Q_MX3000_MY400_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "WkkToWRadionToWWW_M2000_Mr400_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",                  "XToYYprimeTo4Q_MX3000_MY400_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "WkkToWRadionToWWW_M3000_Mr170_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",                  "XToYYprimeTo4Q_MX3000_MY80_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "WkkToWRadionToWWW_M3000_Mr400_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",                  "XToYYprimeTo4Q_MX3000_MY80_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "WkkToWRadionToWWW_M5000_Mr170_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",                  "XToYYprimeTo4Q_MX3000_MY80_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "WkkToWRadionToWWW_M5000_Mr400_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",                  "XToYYprimeTo4Q_MX3000_MY80_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "WpToBpT_Wp2000_Bp170_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lund.h5",             "XToYYprimeTo4Q_MX5000_MY170_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "WpToBpT_Wp2000_Bp25_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lund.h5",              "XToYYprimeTo4Q_MX5000_MY170_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "WpToBpT_Wp2000_Bp400_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lund.h5",             "XToYYprimeTo4Q_MX5000_MY170_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "WpToBpT_Wp2000_Bp80_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lund.h5",              "XToYYprimeTo4Q_MX5000_MY170_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "WpToBpT_Wp3000_Bp170_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lund.h5",             "XToYYprimeTo4Q_MX5000_MY25_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "WpToBpT_Wp3000_Bp25_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lund.h5",              "XToYYprimeTo4Q_MX5000_MY25_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "XToYYprimeTo4Q_MX5000_MY25_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "WpToBpT_Wp3000_Bp80_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lund.h5",              "XToYYprimeTo4Q_MX5000_MY25_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",



        "WpToBpT_Wp5000_Bp170_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lund.h5",             "XToYYprimeTo4Q_MX5000_MY400_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "WpToBpT_Wp5000_Bp25_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lund.h5",              "XToYYprimeTo4Q_MX5000_MY400_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "WpToBpT_Wp5000_Bp400_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lund.h5",             "XToYYprimeTo4Q_MX5000_MY400_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "WpToBpT_Wp5000_Bp80_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lund.h5",              "XToYYprimeTo4Q_MX5000_MY400_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "XToYYprimeTo4Q_MX2000_MY170_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",  "XToYYprimeTo4Q_MX5000_MY80_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "XToYYprimeTo4Q_MX2000_MY170_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",   "XToYYprimeTo4Q_MX5000_MY80_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "XToYYprimeTo4Q_MX2000_MY170_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",  "XToYYprimeTo4Q_MX5000_MY80_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "XToYYprimeTo4Q_MX2000_MY170_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",   "XToYYprimeTo4Q_MX5000_MY80_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "XToYYprimeTo4Q_MX2000_MY25_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",   "YtoHH_Htott_Y2000_H400_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "XToYYprimeTo4Q_MX2000_MY25_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",    "YtoHH_Htott_Y3000_H400_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "XToYYprimeTo4Q_MX2000_MY25_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",   
        "XToYYprimeTo4Q_MX2000_MY25_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",    "ZpToTpTp_Zp2000_Tp400_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "XToYYprimeTo4Q_MX2000_MY400_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",  "ZpToTpTp_Zp3000_Tp400_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "XToYYprimeTo4Q_MX2000_MY400_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",   



        "QstarToQW_M_2000_mW_170_TuneCP2_13TeV-pythia8_TIMBER_Lund.h5",                                
        "QstarToQW_M_2000_mW_25_TuneCP2_13TeV-pythia8_TIMBER_Lund.h5",                                 
        "QstarToQW_M_2000_mW_400_TuneCP2_13TeV-pythia8_TIMBER_Lund.h5",                                
        "QstarToQW_M_2000_mW_80_TuneCP2_13TeV-pythia8_TIMBER_Lund.h5",                                 
        "QstarToQW_M_3000_mW_170_TuneCP2_13TeV-pythia8_TIMBER_Lund.h5",                                
        "QstarToQW_M_3000_mW_25_TuneCP2_13TeV-pythia8_TIMBER_Lund.h5",                                 
        "QstarToQW_M_3000_mW_400_TuneCP2_13TeV-pythia8_TIMBER_Lund.h5",                                
        "QstarToQW_M_3000_mW_80_TuneCP2_13TeV-pythia8_TIMBER_Lund.h5",                                 
        "QstarToQW_M_5000_mW_170_TuneCP2_13TeV-pythia8_TIMBER_Lund.h5",                                
        "QstarToQW_M_5000_mW_25_TuneCP2_13TeV-pythia8_TIMBER_Lund.h5",                                 
        "QstarToQW_M_5000_mW_400_TuneCP2_13TeV-pythia8_TIMBER_Lund.h5",                                
        "QstarToQW_M_5000_mW_80_TuneCP2_13TeV-pythia8_TIMBER_Lund.h5",                                 

        "ZpToTpTp_Zp5000_Tp400_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
        "YtoHH_Htott_Y5000_H400_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",

        #"RSGravitonToGluonGluon_kMpl01_M_1000_TuneCP5_13TeV_pythia8_TIMBER_Lund.h5",                   
        #"RSGravitonToGluonGluon_kMpl01_M_2000_TuneCP5_13TeV_pythia8_TIMBER_Lund.h5",                   
        #"RSGravitonToGluonGluon_kMpl01_M_3000_TuneCP5_13TeV_pythia8_TIMBER_Lund.h5",                   
        #"RSGravitonToGluonGluon_kMpl01_M_5000_TuneCP5_13TeV_pythia8_TIMBER_Lund.h5",                   


        ]


tag = "_july10"

eos_base = "root://cmseos.fnal.gov/"
#eos_base = "/eos/uscms/"
#xrootd_base = "root://cmsxrootd.fnal.gov/"
#xrootd_base = "root://cmsxrootd.fnal.gov/"
xrootd_base = "root://xrootd-cms.infn.it//"

odir = eos_base + "/store/user/oamram/case/sig_files/LundRW/"

memory = 6000

bad_files = []


for sig_name in f_inputs:
    sig_file = odir + sig_name


    label = sig_name.split(".")[0]

    

    cp_sig_cmd = "xrdcp -f %s . \n" % (sig_file)
    cmd = "python jet_images/make_jet_images.py -i %s \n" % sig_name
    cp_cmd = "xrdcp -f %s %s \n" % (sig_name, sig_file)


    script_name = "scripts/script_temp.sh"

    print_and_do("cp scripts/sig_jet_images.sh %s" % script_name)
    f = open(script_name, "a")
    f.write(cp_sig_cmd)
    f.write(cmd)
    f.write(cp_cmd)

    f.close()

    print_and_do("chmod +x %s" % script_name)
    print_and_do("python doCondor.py --sub --njobs 1 --overwrite -s %s -n jet_images_%s "  % (script_name, label + tag))


