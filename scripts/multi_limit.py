from limit_set import *

def print_and_do(s):
    print(s)
    os.system(s)

do_TNT = True
step = 'sys_select'
sig_mass = 2000
mbin = 11

#spbs = " --spbs 10.0 20.0 40.0"
#spbs = " --spbs 30.0 50.0 60.0"
spbs = " --spbs 8.0 12.0 20.0 30.0"
#spbs = " --spbs 30.0 40.0"
#spbs = " --spbs 1.0 4.0 6.0 8.0"
#spbs = " --spbs 0.5 1.0 2.0 4.0"
#spbs = ""
local = False


sig_files = [
#"XToYYprimeTo4Q_MX2000_MY170_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",

#"QstarToQW_M_3000_mW_25_TuneCP2_13TeV-pythia8_TIMBER_Lund.h5",
#"QstarToQW_M_3000_mW_80_TuneCP2_13TeV-pythia8_TIMBER_Lund.h5",
#"QstarToQW_M_3000_mW_170_TuneCP2_13TeV-pythia8_TIMBER_Lund.h5",
"QstarToQW_M_3000_mW_400_TuneCP2_13TeV-pythia8_TIMBER_Lund.h5",
#"WkkToWRadionToWWW_M3000_Mr170_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
"WkkToWRadionToWWW_M3000_Mr400_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"WpToBpT_Wp3000_Bp25_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lund.h5",
#"WpToBpT_Wp3000_Bp80_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lund.h5",
#"WpToBpT_Wp3000_Bp170_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lund.h5",
"WpToBpT_Wp3000_Bp400_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lund.h5",
"ZpToTpTp_Zp3000_Tp400_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
"YtoHH_Htott_Y3000_H400_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",

#"XToYYprimeTo4Q_MX3000_MY25_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX3000_MY25_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX3000_MY25_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX3000_MY25_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX3000_MY80_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX3000_MY80_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX3000_MY80_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX3000_MY170_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX3000_MY170_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
"XToYYprimeTo4Q_MX3000_MY400_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",


#"QstarToQW_M_5000_mW_25_TuneCP2_13TeV-pythia8_TIMBER_Lund.h5",
#"QstarToQW_M_5000_mW_80_TuneCP2_13TeV-pythia8_TIMBER_Lund.h5",
#"QstarToQW_M_5000_mW_170_TuneCP2_13TeV-pythia8_TIMBER_Lund.h5",
#"QstarToQW_M_5000_mW_400_TuneCP2_13TeV-pythia8_TIMBER_Lund.h5",
#"WkkToWRadionToWWW_M5000_Mr170_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"WkkToWRadionToWWW_M5000_Mr400_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"WpToBpT_Wp5000_Bp25_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lund.h5",
#"WpToBpT_Wp5000_Bp80_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lund.h5",
#"WpToBpT_Wp5000_Bp170_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lund.h5",
#"WpToBpT_Wp5000_Bp400_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lund.h5",
#"YtoHH_Htott_Y5000_H400_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"ZpToTpTp_Zp5000_Tp400_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#
#"XToYYprimeTo4Q_MX5000_MY170_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX5000_MY25_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX5000_MY25_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX5000_MY25_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX5000_MY25_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX5000_MY80_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX5000_MY80_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX5000_MY80_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX5000_MY170_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX5000_MY400_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
]

method = 'TNT' if do_TNT else 'cwola'
cmd_base = "python3 limit_set.py "
cmd_new =  cmd_base + "-i ../data/DATA_deta_images/ --data --new --mbin %i --mjj_sig %i " % (mbin, sig_mass)

for fname in sig_files:
    label = fname.split("Tune")[0].replace("_narrow_", "")
    odir = "../runs/limits/%s_%s/" % (method,label)
    if(step == 'new'):
        cmd = cmd_new + spbs + " -o %s --sig_file ../data/%s" % (odir, fname)
        if(do_TNT): 
            cmd += " --do_TNT"
            cmd += " --ae_dir ../models/AEs/AEs_data_SR_june9/"
            if (mbin %10 < 6): cmd += " --condor_mem 2800"
        else:
            cmd += " --condor_mem 1000"
    else:
        cmd = cmd_base + " -o %s --step %s" % (odir, step)
        #if('opt' not in step): cmd += ' --recover'
        if(step == 'train' and len(spbs) > 0): 
            cmd += spbs
    if(local): cmd += " --no-condor"

    if('train' in cmd and do_TNT):
        if (mbin %10 < 6): cmd += " --condor_mem 2800"
    print_and_do(cmd)

    if(step == 'output'):
        if(do_TNT): summary_dir = '../runs/limits/summary_TNT/'
        else: summary_dir = '../runs/limits/summary_cwola/'

        json_name = fname.replace("_Lund.h5", ".json")

        print_and_do("cp " + odir + json_name + " " + summary_dir + json_name)

