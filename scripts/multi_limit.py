from limit_set import *

def print_and_do(s):
    print(s)
    os.system(s)

do_TNT = True
#steps = ['merge', 'sys_merge', 'plot', 'sys_plot', 'output']
#steps = ['sys_train', 'sys_select']
#steps = ['sys_merge', 'sys_plot', 'output']
#steps = ['plot', 'sys_plot']
steps = ['output', 'clean']
#steps = ['output']
sig_mass = 3000
mbin = 13
#mbin = 6

#spbs = " --spbs 10.0 20.0 40.0"
#spbs = " --spbs 20.0 30.0 40.0"
spbs = " --spbs 12.0 20.0 30.0"
#spbs = " --spbs 8.0 12.0 20.0 30.0"
#spbs = " --spbs 40 50"
#spbs = " --spbs 2 4 6"
#spbs = ""
inc_spb = False
local = False


sig_files = [
#"XToYYprimeTo4Q_MX2000_MY170_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",

#"QstarToQW_M_3000_mW_25_TuneCP2_13TeV-pythia8_TIMBER_Lund.h5",
#"QstarToQW_M_3000_mW_80_TuneCP2_13TeV-pythia8_TIMBER_Lund.h5",
#"QstarToQW_M_3000_mW_170_TuneCP2_13TeV-pythia8_TIMBER_Lund.h5",
#"QstarToQW_M_3000_mW_400_TuneCP2_13TeV-pythia8_TIMBER_Lund.h5",
#"WkkToWRadionToWWW_M3000_Mr170_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"WkkToWRadionToWWW_M3000_Mr400_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"WpToBpT_Wp3000_Bp25_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lund.h5",
#"WpToBpT_Wp3000_Bp80_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lund.h5",
#"WpToBpT_Wp3000_Bp170_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lund.h5",
#"WpToBpT_Wp3000_Bp400_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lund.h5",
"ZpToTpTp_Zp3000_Tp400_TuneCP5_13TeV-madgraph-pythia8_FILTERED_TIMBER_Lund.h5",
#"YtoHH_Htott_Y3000_H400_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",

#"XToYYprimeTo4Q_MX3000_MY25_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX3000_MY25_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX3000_MY25_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX3000_MY25_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX3000_MY80_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX3000_MY80_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX3000_MY80_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX3000_MY170_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX3000_MY170_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX3000_MY400_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",


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
##"ZpToTpTp_Zp5000_Tp400_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#
#"XToYYprimeTo4Q_MX5000_MY25_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX5000_MY25_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX5000_MY25_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX5000_MY25_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX5000_MY80_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX5000_MY80_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX5000_MY80_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX5000_MY170_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX5000_MY170_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
#"XToYYprimeTo4Q_MX5000_MY400_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lund.h5",
]

method = 'TNT' if do_TNT else 'cwola'
cmd_base = "python3 limit_set.py "
cmd_new =  cmd_base + "-i ../data/DATA_deta_images/ --data --new --mbin %i --mjj_sig %i " % (mbin, sig_mass)

for step in steps:
    for fname in sig_files:
        label = fname.split("Tune")[0].replace("_narrow_", "")
        odir = "../runs/limits/%s_%s/" % (method,label)
        if(step == 'new'):
            cmd = cmd_new + spbs + " -o %s --sig_file ../data/%s" % (odir, fname)
            if(do_TNT): 
                cmd += " --do_TNT"
                cmd += " --ae_dir ../models/AEs/AEs_data_SR_june9/"
            else:
                cmd += " --condor_mem 1000"
        else:
            cmd = cmd_base + " -o %s --step %s" % (odir, step)
            #if('opt' not in step): cmd += ' --recover'
            if((step == 'train' and len(spbs) > 0) or inc_spb): 
                cmd += spbs
        if(local): cmd += " --no-condor"

        if(('train' in cmd or 'new' in cmd) and do_TNT):
            cmd += ' --saved_AE_scores'
            #if (mbin == 13): cmd += " --condor_mem 2800"
            #if (mbin == 11): cmd += " --condor_mem 2800"
        print_and_do(cmd)

        if(step == 'output'):
            if(do_TNT): summary_dir = '../runs/limits/summary_TNT_v2'
            else: summary_dir = '../runs/limits/summary_cwola_v2'

            json_name = fname.replace("_Lund.h5", ".json")

            print_and_do("cp " + odir + json_name + " " + summary_dir + '_nosys/' + json_name)

            if(os.path.exists(odir + json_name + '.sys')):
                print_and_do("cp " + odir + json_name + ".sys " + summary_dir + '/' + json_name)

            if(os.path.exists(odir + json_name + '.signifs')):
                print_and_do("cp " + odir + json_name + ".signifs " + summary_dir + '_signifs' + '/' + json_name)

            if(os.path.exists(odir + json_name + '.inc')):
                print_and_do("cp " + odir + json_name + ".inc " + ' ../runs/limits/summary_inclusive/' + json_name)


