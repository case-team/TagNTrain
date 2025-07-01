import sys, commands, os, fnmatch

def print_and_do(s):
    print(s)
    return os.system(s)


inputs = [

        #2016APV

    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2016APV/WJetsToLNu/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-100To200/211106_195837/0000/', 'WJets_HT100to200'),
    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2016APV/WJetsToLNu/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-200To400/211106_195956/0000/', 'WJets_HT200to400'),
    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2016APV/WJetsToLNu/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-400To600/211106_200039/0000/', 'WJets_HT400to600'),
    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2016APV/WJetsToLNu/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-600To800/211106_200016/0000/', 'WJets_HT600to800'),
    #('/eos/uscms//store/user/lpcpfnano/drankin/v2_2/2016APV/WJetsToLNu/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-800To1200/211106_195936/0000/', 'WJets_HT800to1200'),
    #('/eos/uscms//store/user/lpcpfnano/drankin/v2_2/2016APV/WJetsToLNu/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-1200To2500/211106_195856/0000/', 'WJets_HT1200to2500'),
    #('/eos/uscms//store/user/lpcpfnano/drankin/v2_2/2016APV/WJetsToLNu/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-2500ToInf/211106_195916/0000/', 'WJets_HT2500toInf'),


    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2016APV/TTbar/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/TTToSemiLeptonic_ext1/211112_131727/0000/', 'TTToSemiLeptonic_large'),
    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2016APV/TTbar/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/TTToHadronic/211112_131700/0000/', 'TTToHadronic'),
    #('/eos/uscms/store/user/lpcpfnano/yihan/v2_2/2016APV/TTbar/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/TTTo2L2Nu/211106_012850/0000/', 'TTTo2L2Nu'),

    #('/eos/uscms/store/user/lpcpfnano/jdickins/v2_2/2016APV/Diboson/WW_TuneCP5_13TeV-pythia8/WW/220310_211306/0000/', 'WW'),
    #('/eos/uscms/store/user/lpcpfnano/jdickins/v2_2/2016APV/Diboson/WZ_TuneCP5_13TeV-pythia8/WZ/220310_211323/0000/', 'WZ'),
    #('/eos/uscms/store/user/lpcpfnano/jdickins/v2_2/2016APV/Diboson/ZZ_TuneCP5_13TeV-pythia8/ZZ/220310_211340/0000/', 'ZZ'),


    #('/eos/uscms/store/user/lpcpfnano/cmantill/v2_3/2016APV/SingleTop/ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/ST_tW_antitop_5f_NoFullyHadronicDecays/220808_173508/0000/',
    #    'SingleTop_antitW'),
    #('/eos/uscms/store/user/lpcpfnano/cmantill/v2_3/2016APV/SingleTop/ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/ST_tW_top_5f_NoFullyHadronicDecays/220808_173419/0000/',
    #    'SingleTop_tW'),
    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2016APV/SingleTop/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8/ST_s-channel_4f_leptonDecays/220601_184931/0000/',
    #    'SingleTop_SChan'),

    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2016APV/SingleTop/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/ST_t-channel_antitop_4f_InclusiveDecays/220601_185020/0000/',
    #    'SingleAntiTop'),
    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2016APV/SingleTop/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/ST_t-channel_top_4f_InclusiveDecays/220601_185110/0000/', 'SingleTop'),


        #2016

    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2016/WJetsToLNu/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-100To200/211106_175049/0000/', 'WJets_HT100to200'),
    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2016/WJetsToLNu/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-200To400/211106_175321/0000/', 'WJets_HT200to400'),
    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2016/WJetsToLNu/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-400To600/211106_175440/0000/', 'WJets_HT400to600'),
    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2016/WJetsToLNu/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-600To800/211106_175401/0000/', 'WJets_HT600to800'),
    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2016/WJetsToLNu/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-800To1200/211106_175255/0000/', 'WJets_HT800to1200'),
    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2016/WJetsToLNu/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-1200To2500/211106_175110/0000/', 'WJets_HT1200to2500'),
    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2016/WJetsToLNu/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-2500ToInf/211106_175341//0000/', 'WJets_HT2500toInf'),


    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2016/TTbar/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/TTToSemiLeptonic_ext1/211112_131315/0000/', 'TTToSemiLeptonic_large'),
    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2016/TTbar/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/TTToHadronic/211112_131246/0000/', 'TTToHadronic'),
    #('/eos/uscms/store/user/lpcpfnano/yihan/v2_2/2016/TTbar/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/TTTo2L2Nu/211106_011324/0000/', 'TTTo2L2Nu'),

    #('/eos/uscms/store/user/lpcpfnano/jdickins/v2_2/2016/Diboson/WW_TuneCP5_13TeV-pythia8/WW/220310_204910/0000/', 'WW'),
    #('/eos/uscms/store/user/lpcpfnano/jdickins/v2_2/2016/Diboson/WZ_TuneCP5_13TeV-pythia8/WZ/220310_204927/0000/', 'WZ'),
    #('/eos/uscms/store/user/lpcpfnano/jdickins/v2_2/2016/Diboson/ZZ_TuneCP5_13TeV-pythia8/ZZ/220310_204944/0000/', 'ZZ'),


    #('/eos/uscms/store/user/lpcpfnano/pharris/v2_2/2016/SingleTop/ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/ST_tW_antitop_5f_NoFullyHadronicDecays/211115_040641/0000/',
    #    'SingleTop_antitW'),
    #('/eos/uscms/store/user/lpcpfnano/pharris/v2_2/2016/SingleTop/ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/ST_tW_top_5f_NoFullyHadronicDecays/211115_040546/0000/',
    #    'SingleTop_tW'),
    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2016/SingleTop/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8/ST_s-channel_4f_leptonDecays/220601_161223/0000/',
    #    'SingleTop_SChan'),

    #('/eos/uscms/store/user/lpcpfnano/pharris/v2_2/2016/SingleTop/ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8/ST_t-channel_antitop_5f_InclusiveDecays/211115_040707/0000/',
    #    'SingleAntiTop'),
    #('/eos/uscms/store/user/lpcpfnano/pharris/v2_2/2016/SingleTop/ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8/ST_t-channel_top_5f_InclusiveDecays/211115_040410/0000/',
    #    'SingleTop'),




        #2017
    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2017/WJetsToLNu/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-100To200/211106_200115/0000/', 'WJets_HT100to200'),
    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2017/WJetsToLNu/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-200To400/211106_200215/0000/', 'WJets_HT200to400'),
    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2017/WJetsToLNu/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-400To600/211106_200335/0000/', 'WJets_HT400to600'),
    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2017/WJetsToLNu/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-600To800/211106_200255/0000/', 'WJets_HT600to800'),
    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2017/WJetsToLNu/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-800To1200/211106_200154/0000/', 'WJets_HT800to1200'),
    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2017/WJetsToLNu/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-1200To2500/211106_200134/0000/', 'WJets_HT1200to2500'),
    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2017/WJetsToLNu/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-2500ToInf/211106_200235/0000/', 'WJets_HT2500toInf'),


    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2017/TTbar/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/TTToSemiLeptonic_ext1/211112_132937/0000/', 'TTToSemiLeptonic_large'),
    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2017/TTbar/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/TTToHadronic/211112_132910/0000/', 'TTToHadronic'),
    #('/eos/uscms/store/user/lpcpfnano/yihan/v2_2/2017/TTbar/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/TTTo2L2Nu/211105_112519/0000/', 'TTTo2L2Nu'),

    #('/eos/uscms/store/user/lpcpfnano/jdickins/v2_2/2017/Diboson/WW_TuneCP5_13TeV-pythia8/WW/220310_210655/0000/', 'WW'),
    #('/eos/uscms/store/user/lpcpfnano/jdickins/v2_2/2017/Diboson/WZ_TuneCP5_13TeV-pythia8/WZ/220310_210712/0000/', 'WZ'),
    #('/eos/uscms/store/user/lpcpfnano/jdickins/v2_2/2017/Diboson/ZZ_TuneCP5_13TeV-pythia8/ZZ/220310_210733/0000/', 'ZZ'),

    #('/eos/uscms/store/user/lpcpfnano/pharris/v2_2/2017/SingleTop/ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/ST_tW_antitop_5f_NoFullyHadronicDecays/211115_041345/0000/',
    #    'SingleTop_antitW'),
    #('/eos/uscms/store/user/lpcpfnano/pharris/v2_2/2017/SingleTop/ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/ST_tW_top_5f_NoFullyHadronicDecays/211115_041213/0000/',
    #    'SingleTop_tW'),
    #('/eos/uscms/store/user/lpcpfnano/pharris/v2_2/2017/SingleTop/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8/ST_s-channel_4f_leptonDecays/211115_041411/0000/',
    #    'SingleTop_SChan'),

    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2017/SingleTop/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/ST_t-channel_antitop_4f_InclusiveDecays/220602_121529/0000/',
    # 'SingleAntiTop'),
    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2017/SingleTop/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/ST_t-channel_top_4f_InclusiveDecays/220602_121553/0000/',
     #'SingleTop'),




    #2018
    #('/store/user/lpcpfnano/drankin/v2_2/2018/WJetsToLNu/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-100To200/211107_161635/0000/', 'WJets_HT100to200'),
    #('/store/user/lpcpfnano/drankin/v2_2/2018/WJetsToLNu/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-200To400/211107_161736/0000/', 'WJets_HT200to400'),
    #('/store/user/lpcpfnano/drankin/v2_2/2018/WJetsToLNu/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-400To600/211107_161835/0000/', 'WJets_HT400to600'),
    #('/store/user/lpcpfnano/drankin/v2_2/2018/WJetsToLNu/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-600To800/211107_161756/0000/', 'WJets_HT600to800'),
    #('/store/user/lpcpfnano/drankin/v2_2/2018/WJetsToLNu/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-800To1200/211107_161715/0000/', 'WJets_HT800to1200'),
    #('/store/user/lpcpfnano/drankin/v2_2/2018/WJetsToLNu/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-1200To2500/211107_161656/0000/', 'WJets_HT1200to2500'),
    #('/store/user/lpcpfnano/drankin/v2_2/2018/WJetsToLNu/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToLNu_HT-2500ToInf/211115_190243/0000/', 'WJets_HT2500toInf'),


    ##('/eos/uscms/store/user/oamram/case/2018/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer19UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1_PFNano/220901_212212/0000/',
    #    #'TTToSemiLeptonic'),
    #('/eos/uscms/store/user/lpcpfnano/emoreno/v2_2/2018/TTbar/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/TTToSemiLeptonic/211208_201500/0000/', 'TTToSemiLeptonic_large'),
    #('/eos/uscms/store/user/lpcpfnano/drankin/v2_2/2018/TTbar/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/TTToHadronic/211112_133103/0000/', 'TTToHadronic'),
    #('/eos/uscms/store/user/lpcpfnano/yihan/v2_2/2018/TTbar/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/TTTo2L2Nu/211106_013754/0000/', 'TTTo2L2Nu'),

    #('/eos/uscms/store/user/lpcpfnano/jdickins/v2_2/2018/Diboson/WW_TuneCP5_13TeV-pythia8/WW/220310_201228/0000/', 'WW'),
    #('/eos/uscms/store/user/lpcpfnano/jdickins/v2_2/2018/Diboson/WZ_TuneCP5_13TeV-pythia8/WZ/220310_201245/0000/', 'WZ'),
    #('/eos/uscms/store/user/lpcpfnano/jdickins/v2_2/2018/Diboson/ZZ_TuneCP5_13TeV-pythia8/ZZ/220314_153133/0000/', 'ZZ'),

    #('/eos/uscms/store/user/lpcpfnano/pharris/v2_2/2018/SingleTop/ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/ST_tW_antitop_5f_NoFullyHadronicDecays/211115_042731/0000/',
    #    'SingleTop_antitW'),
    #('/eos/uscms/store/user/lpcpfnano/pharris/v2_2/2018/SingleTop/ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/ST_tW_top_5f_NoFullyHadronicDecays/211115_042828/0000/',
    #    'SingleTop_tW'),
    #('/eos/uscms/store/user/lpcpfnano/pharris/v2_2/2018/SingleTop/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8/ST_s-channel_4f_leptonDecays/211115_042757/0000/',
    #    'SingleTop_SChan'),

    #('/store/user/lpcpfnano/drankin/v2_2/2018/SingleTop/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/ST_t-channel_antitop_4f_InclusiveDecays/220602_122533/0000/',
    # 'SingleAntiTop'),
    #('/store/user/lpcpfnano/drankin/v2_2/2018/SingleTop/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/ST_t-channel_top_4f_InclusiveDecays/220602_122557/0000/',
    # 'SingleTop'),

    ('/store/user/oamram/case/2018/TT_TuneCH3_13TeV-powheg-herwig7/RunIISummer20UL18MiniAOD-106X_upgrade2018_realistic_v16_L1v1-v1_PFNano/240312_235833/0000/', 'TT_herwig_reco'),

]
#inputs = [('/store/user/oamram/test', 'Nano_postprocess_test')]

max_files = -1

tag = "_may2"


eos_base = "root://cmseos.fnal.gov/"
temp_eos_dir = "/store/user/oamram/case/Lund_files/2018/Skimmed/"
#outdir = "./"

hadd = False




for eos_dir, label in inputs:

    print("mkdir /eos/uscms/" + temp_eos_dir + label)
    os.system("mkdir /eos/uscms/" + temp_eos_dir + label)

    outdir  = eos_base + temp_eos_dir + label

    if(not hadd):
        cmd = "python postprocess.py -i %s -l %s --max_files %i --outdir %s \n" % (eos_dir, label, max_files, outdir)
        cp_cmd = "xrdcp -f %s/hadd_%s.root ${1} \n" % (outdir, label)

        script_name = "scripts/script_temp.sh"
        print_and_do("cp scripts/postprocess_template.sh %s" % script_name)
        f = open(script_name, "a")
        f.write(cmd)
        #f.write(cp_cmd)
        f.close()
        print_and_do("chmod +x %s" % script_name)
        print_and_do("python doCondor.py --njobs 1  --overwrite --cmssw --sub  -s %s -n Nano_postprocess_%s --mem 6000 "  % (script_name, label + tag))
        #print_and_do("rm %s" % script_name)
    else:

        tmp_hadd_dir  = "/storage/local/data1/gpuscratch/oamram/temp/"
        hadd_out = tmp_hadd_dir + label + "_hadd.root"


        cmd = "ls -d /eos/uscms/%s/* > temp.txt" % (temp_eos_dir + label)
        os.system(cmd)
        print(cmd)
        f = open("temp.txt")
        fnames = f.read().splitlines()

        if(len(fnames) > max_files):
            print("Reducing down to %i files" % max_files)
            for f in fnames[max_files:]:
                print("eos rm %s" % f)
                os.system("eos rm %s" % f)
            fnames = fnames[:max_files]

        hadd_cmd = "hadd -f " + hadd_out 

        for name in fnames:
            hadd_cmd += " " + eos_base + name

        print(hadd_cmd)
        os.system(hadd_cmd)
        
