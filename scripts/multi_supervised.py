import os
import json

def print_and_do(s):
    print(s)
    os.system(s)



#sig_mass = 5000
#mbin = 6
sig_mass = 3000
mbin = 13
num_epoch = 5
train = False
ATLAS = True
plot = True
fout_name = "../plots/supervised_classifiers_all/TNT_5000.json"

sig_files = [

#"QstarToQW_M_3000_mW_25_TuneCP2_13TeV-pythia8_TIMBER_Lundv2.h5",
#"QstarToQW_M_3000_mW_80_TuneCP2_13TeV-pythia8_TIMBER_Lundv2.h5",
#"QstarToQW_M_3000_mW_170_TuneCP2_13TeV-pythia8_TIMBER_Lundv2.h5",
#"QstarToQW_M_3000_mW_400_TuneCP2_13TeV-pythia8_TIMBER_Lundv2.h5",
#"WkkToWRadionToWWW_M3000_Mr170_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lundv2.h5",
"WkkToWRadionToWWW_M3000_Mr400_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lundv2.h5",
#"WpToBpT_Wp3000_Bp25_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lundv2.h5",
#"WpToBpT_Wp3000_Bp80_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lundv2.h5",
#"WpToBpT_Wp3000_Bp170_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lundv2.h5",
"WpToBpT_Wp3000_Bp400_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lundv2.h5",
#"ZpToTpTp_Zp3000_Tp400_TuneCP5_13TeV-madgraph-pythia8_FILTERED_TIMBER_Lundv2.h5",
"YtoHH_Htott_Y3000_H400_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lundv2.h5",
#
#"XToYYprimeTo4Q_MX3000_MY25_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lundv2.h5",
#"XToYYprimeTo4Q_MX3000_MY25_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lundv2.h5",
#"XToYYprimeTo4Q_MX3000_MY25_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lundv2.h5",
#"XToYYprimeTo4Q_MX3000_MY25_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lundv2.h5",
#"XToYYprimeTo4Q_MX3000_MY80_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lundv2.h5",
#"XToYYprimeTo4Q_MX3000_MY80_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lundv2.h5",
#"XToYYprimeTo4Q_MX3000_MY80_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lundv2.h5",
"XToYYprimeTo4Q_MX3000_MY170_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lundv2.h5",
#"XToYYprimeTo4Q_MX3000_MY170_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lundv2.h5",
"XToYYprimeTo4Q_MX3000_MY400_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lundv2.h5",


#"QstarToQW_M_5000_mW_25_TuneCP2_13TeV-pythia8_TIMBER_Lundv2.h5",
#"QstarToQW_M_5000_mW_80_TuneCP2_13TeV-pythia8_TIMBER_Lundv2.h5",
#"QstarToQW_M_5000_mW_170_TuneCP2_13TeV-pythia8_TIMBER_Lundv2.h5",
#"QstarToQW_M_5000_mW_400_TuneCP2_13TeV-pythia8_TIMBER_Lundv2.h5",
#"WkkToWRadionToWWW_M5000_Mr170_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lundv2.h5",
#"WkkToWRadionToWWW_M5000_Mr400_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lundv2.h5",
#"WpToBpT_Wp5000_Bp25_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lundv2.h5",
#"WpToBpT_Wp5000_Bp80_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lundv2.h5", 
#"WpToBpT_Wp5000_Bp170_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lundv2.h5", 
#"WpToBpT_Wp5000_Bp400_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER_Lundv2.h5",
#"YtoHH_Htott_Y5000_H400_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lundv2.h5",
#
#"XToYYprimeTo4Q_MX5000_MY25_MYprime25_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lundv2.h5",
#"XToYYprimeTo4Q_MX5000_MY25_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lundv2.h5",
#"XToYYprimeTo4Q_MX5000_MY25_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lundv2.h5",
#"XToYYprimeTo4Q_MX5000_MY25_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lundv2.h5",
#"XToYYprimeTo4Q_MX5000_MY80_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lundv2.h5",
#"XToYYprimeTo4Q_MX5000_MY80_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lundv2.h5",
#"XToYYprimeTo4Q_MX5000_MY80_MYprime80_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lundv2.h5",
#"XToYYprimeTo4Q_MX5000_MY170_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lundv2.h5",
#"XToYYprimeTo4Q_MX5000_MY170_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lundv2.h5",
#"XToYYprimeTo4Q_MX5000_MY400_MYprime400_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lundv2.h5",
]

if(train):
    for fname in sig_files:
        sig_name = fname.replace("_Lundv2.h5", "")
        print(sig_name)

        if(not ATLAS):
            print_and_do(f"mkdir ../plots/supervised_classifiers_all/{sig_name}/")
            cmd1 = f"python3 ../training/train_supervised_network.py -i ../data/DATA_deta_images/ --batch_start 0 --batch_stop 25  --val_batch_start 25 --val_batch_stop 30 --sig_file ../data/LundRW/{fname}  --data  --mbin {mbin} --num_epoch {num_epoch} -o ../plots/supervised_classifiers_all/{sig_name}/j1.h5 -j 1"

            cmd2 = f"python3 ../training/train_supervised_network.py -i ../data/DATA_deta_images/ --batch_start 0 --batch_stop 25  --val_batch_start 25 --val_batch_stop 30 --sig_file ../data/LundRW/{fname}  --data  --mbin {mbin} --num_epoch {num_epoch} -o ../plots/supervised_classifiers_all/{sig_name}/j2.h5 -j 2"

            print_and_do(cmd1)
            print_and_do(cmd2)
        else:
            print_and_do(f"mkdir -p ../plots/supervised_classifiers_ATLAS/{sig_name}/")
            cmd1 = f"python3 ../training/train_supervised_network.py -i ../data/DATA_deta_images/ --batch_start 0 --batch_stop 25  --val_batch_start 25 --val_batch_stop 30 --sig_file ../data/LundRW/{fname}  --data  --mbin {mbin} --num_epoch {num_epoch} -o ../plots/supervised_classifiers_ATLAS/{sig_name}/jj.h5 --use_both --m_only_feats --no_ptrw --small_net"

            print_and_do(cmd1)

elif(plot):
    for fname in sig_files:
        sig_name = fname.replace("_Lundv2.h5", "")

        if(not ATLAS):
            cmd = "python3 ../plotting/roc_event.py -i ../data/DATA_deta_images/ --batch_start 31 --batch_stop 39 --sig_file ../data/LundRW/%s  --data  --mbin %i -o ../plots/supervised_classifiers_all/%s/ --labeler_name  ../plots/supervised_classifiers_all/%s/{j_label}.h5" %(fname, mbin, sig_name, sig_name)
        else:
            cmd = "python3 ../plotting/roc_event.py -i ../data/DATA_deta_images/ --batch_start 31 --batch_stop 39 --sig_file ../data/LundRW/%s  --data  --mbin %i -o ../plots/supervised_classifiers_ATLAS/%s/ --labeler_name  ../plots/supervised_classifiers_ATLAS/%s/jj.h5 --m_only_feats --small_net" %(fname, mbin, sig_name, sig_name)

        print_and_do(cmd)
else:
    all_metrics = {}
    for fname in sig_files:
        sig_name = fname.replace("_Lundv2.h5", "")
        json_name = "../plots/supervised_classifiers_all/%s/metrics.json" % sig_name
        
        with open(json_name, "rb") as f:
            metrics = json.load(f )
            all_metrics.update(metrics)

    print("writing to " + fout_name)
    print(all_metrics)
    with open(fout_name, "w") as f:
        json.dump(all_metrics, f )

#python3 roc_event.py -i ../data/DATA_deta_images/ --batch_start 31 --batch_stop 39 --sig_file ../data/LundRW/XToYYprimeTo4Q_MX3000_MY170_MYprime170_narrow_TuneCP5_13TeV-madgraph-pythia8_TIMBER_Lundv2.h5 -o ../plots/supervised_classifiers/XYY_5TeV/ --data --mbin 6 --no_lund_weights

