import sys
import os
sys.path.append('..')
from utils.TrainingUtils import *
from plotting.model_interpretation import model_interp
from create_model_ensemble import *
from selection import *
import subprocess
import h5py
import time

fit_cmd_setup = "cd ../fitting; source deactivate; eval `scramv1 runtime -sh`;"  
fit_cmd_after = "cd -; source deactivate; source activate mlenv0"

def run_dijetfit(options, fit_start = -1, fit_stop = -1, sig_shape_file = "", input_file = "", label = "", output_dir = "", loop = False):
    print("dijet fit MJJ sig %.1f"  % options.mjj_sig)
    base_path = os.path.abspath(".") + "/"
    if(len(output_dir) == 0): output_dir = base_path + options.output
    if(len(input_file) == 0): 
        fit_inputs = 'fit_inputs_eff{eff}.h5'.format(eff = options.effs[0])
        input_file = base_path + options.output + fit_inputs



    ftest_thresh = 0.1
    err_thresh = 0.15
    dijet_cmd = "python dijetfit.py -i %s -p %s --rebin --ftest_thresh %.2f --err_thresh %.2f" % (input_file, output_dir, ftest_thresh, err_thresh)
    if('sig_norm_unc' in options.__dict__.keys() and options.sig_norm_unc > 0.):
        dijet_cmd += " --sig_norm_unc %.3f " % options.sig_norm_unc
    if(len(sig_shape_file) > 0):
        dijet_cmd +=  " -M %.0f --sig_shape %s --dcb-model -l %sM%0.f" % (options.mjj_sig, sig_shape_file,  label, options.mjj_sig)

    if(options.mjj_sig > 4500):
        #choose more realistic normalization for high mass signals
        dijet_cmd += " --sig_norm 10.0"




    run_fit = True
    last_change = 'start'
    while(run_fit):
        dijet_cmd_iter = copy.copy(dijet_cmd)

        if(fit_start > 0):
            dijet_cmd_iter += " --mjj_min %.0f" % fit_start

        if(fit_stop > 0):
            dijet_cmd_iter += " --mjj_max %.0f" % fit_stop

        dijet_cmd_iter += " >& %s/fit_log_%s%.1f.txt " % (output_dir, label, options.mjj_sig)
        fit_cmd = fit_cmd_setup +  dijet_cmd_iter + "; " + fit_cmd_after
        print(fit_cmd)
        subprocess.call(fit_cmd,  shell = True, executable = '/bin/bash')

        run_fit = False
        no_results = False
        if(loop): #don't stop until we get a good fit
            fit_file = output_dir + 'fit_results_%.1f.json' % options.mjj_sig
            if(not os.path.exists(fit_file)):
                print("\nFit didn't converge")
                no_results = True
                run_fit = True
            else:
                with open(fit_file, 'r') as f:
                    fit_params = json.load(f, encoding="latin-1")

                if((fit_params['bkgfit_prob'] < 0.05 and fit_params['sbfit_prob'] < 0.05) or fit_params['bkgfit_frac_err'] > err_thresh):
                #if((fit_params['bkgfit_prob'] < 0.05) or fit_params['bkgfit_frac_err'] > err_thresh):
                    run_fit = True
                    print("\nPOOR Fit quality (bkg pval %.2e, s+b pval %.2e, fit unc %.2f)!" % 
                            (fit_params['bkgfit_prob'], fit_params['sbfit_prob'], fit_params['bkgfit_frac_err']))

            if(run_fit):
                if(fit_start < 1550.): fit_start = 1550.
                elif(fit_stop < 0 or fit_stop > 6500.): fit_stop = 6500.
                elif(last_change == "end" and (fit_start + 400. <= options.mjj_sig) ): 
                    fit_start += 100.
                    last_change = 'start'
                elif(fit_stop - 500. >= options.mjj_sig): 
                    if(fit_stop > 5000): fit_stop -= 500
                    else: fit_stop -= 200.
                    last_change = 'end'
                elif((fit_start + 250. <= options.mjj_sig) ): 
                    fit_start += 100.
                else:
                    print("No boundary changes found!")
                    run_fit = False

                print("Changing fit start/stop to %.0f/%.0f and retrying" % (fit_start, fit_stop))
    return fit_start



def check_all_models_there(model_dir, num_models):
    #TODO Fix it so there aren't Nan's...
    print("Checking %s" % model_dir)
    missing_models = []
    files = os.listdir(model_dir)
    for i in range(num_models):
        model_name = "model%i.h5" % i
        if(model_name not in files):
            print("Missing %s " % model_name)
            missing_models.append(i)

    if(len(missing_models) == num_models):
        print("Missing all models! Something wrong in training?? This is bad !!!")
        sys.exit(1)
    else:
        for i in missing_models:
            cpy_i = (i + 1) % num_models
            for idx in range(num_models):
                if(cpy_i not in missing_models): break
                else:
                    cpy_i = (cpy_i +1) % num_models

            cpy_name = "model%i.h5" % cpy_i
            dest_name = "model%i.h5" % i
            cpy_cmd = "cp %s %s" % (model_dir + cpy_name, model_dir + dest_name)
            print("Copying : " + cpy_cmd)
            os.system(cpy_cmd)



def avg_eff(fout_name, input_list):
    if(len(input_list) == 0): return
    all_effs = dict()
    for fname in input_list:
        f = h5py.File(fname, "r")

        for key in f.keys():
            if("eff" not in key): continue

            if(key in all_effs.keys()):
                all_effs[key].append(f[key][0])
            else:
                all_effs[key] = [f[key][0]]

    avg_effs = dict()

    fout = h5py.File(fout_name, "a")
    for key,val in all_effs.items():
        avg_eff = np.mean(val)
        #print(key, avg_eff)
        if(key not in fout.keys()):
            fout.create_dataset(key, data=np.array([avg_eff]) )
        else:
            fout[key][0] = avg_eff








def full_run(options):

    #if(len(options.opts) > 0):
    #    options = get_options_from_json(options.opts)

    if(options.output[-1] != '/'):
        options.output += '/'

    if(len(options.label) == 0):
        if(options.output[-1] == "/"):
            options.label = options.output.split("/")[-2]
        else:
            options.label = options.output.split("/")[-1]

    if(not os.path.exists(options.output)):
        subprocess.call("mkdir %s" % options.output, shell = True)

    if(options.reload):
        if(os.path.exists(options.output + "run_opts.json")):
            rel_opts = get_options_from_json(options.output + "run_opts.json")
        elif(os.path.exists(options.output + "run_opts.pkl")):
            rel_opts = get_options_from_pkl(options.output + "run_opts.pkl")
        else:
            print("Reload options specified but file %s doesn't exist. Exiting" % (options.output+"run_opts.pkl"))
            sys.exit(1)

        rel_opts.step = options.step
        if(len(options.effs) >0): rel_opts.effs = options.effs
        if(options.sig_per_batch >= 0): rel_opts.sig_per_batch = options.sig_per_batch
        if('fit_start' in options.__dict__.keys() and options.fit_start > 0): rel_opts.fit_start = options.fit_start
        rel_opts.condor = options.condor
        if('fit_label' in options.__dict__.keys()): rel_opts.fit_label = options.fit_label
        if('generic_sig_shape' in options.__dict__.keys()): rel_opts.generic_sig_shape = options.generic_sig_shape
        rel_opts.keep_LSF = options.keep_LSF #quick fix
        rel_opts.redo_roc = options.redo_roc #quick fix
        rel_opts.condor_mem = options.__dict__.get("condor_mem", -1)
        rel_opts.recover = options.recover
        rel_opts.lund_weights = options.__dict__.get('lund_weights', True)
        rel_opts.sig_shape = options.__dict__.get('sig_shape', "")
        if(abs(options.mjj_sig - 2500.) > 1.):  rel_opts.mjj_sig  = options.mjj_sig #quick fix
        rel_opts.output = options.output #allows renaming / moving directories
        options = rel_opts
    else:
        #save options
        options_dict = options.__dict__
        write_options_to_json(options_dict, options.output + "run_opts.json" )


 
    if(options.do_TNT):
        options.randsort = True
        options.score_comb = "mult"
    else:
        options.score_comb = "max"

    print("Condor %i" % options.condor)


    if(options.numBatches % options.kfolds != 0):
        print("Number of batches (%i) must be multiple of number of kfolds (%i)" % (options.numBatches, options.kfolds))
        sys.exit(1)

    ksize = options.numBatches // options.kfolds

    batches_per_kfold = (options.kfolds -1 ) * ksize
    if(batches_per_kfold % options.lfolds !=0):
        print("Number of batches per kfold(%i) must be multiple of number of lfolds (%i)" % (batches_per_kfold, options.lfolds))
        sys.exit(1)

    if(options.step not in ["train", "get", "select", "merge", "fit", "roc", "bias", "interp", "clean", "all"]):
        print("Invalid option %s" % options.step)
        sys.exit(1)

    #parse what to do 
    get_condor = do_train = do_merge = do_selection = do_fit = False
    do_train = options.step == "train"
    get_condor = options.step == "get"
    do_selection = options.step == "select"
    do_merge = options.step == "merge"
    do_fit = options.step == "fit"
    do_roc = options.step == "roc"
    do_interp = options.step == "interp"
    do_bias_test = options.step == "bias"
    do_clean = options.step == "clean"

    if(options.step == "all"):
        do_train = do_selection = do_fit = do_roc = True
    if(options.step == "roc" and options.condor):
        do_merge = True

    #get_condor = get_condor or (do_selection and options.condor)
    do_merge = do_merge 


    options.num_val_batch = batches_per_kfold // options.lfolds

    total_batch_list = list(range(options.numBatches))

    base_path = os.path.abspath(".") + "/"

    start_time = time.time()
    kfold_options = []

    output = None

    if(not options.condor and not(os.path.exists(options.sig_file))):
        options.sig_file = options.sig_file.replace("data", "data/LundRW")


    for k in range(options.kfolds):

        k_options = copy.deepcopy(options)
        k_list = copy.copy(total_batch_list)

        k_options.holdouts = list(range(k* ksize, (k+1) * ksize))

        for to_remove in k_options.holdouts:
            k_list.remove(to_remove)


        k_options.data_batch_list = k_list

        kfold_options += [k_options]



    #Do trainings
    if(do_train):
        for k,k_options in enumerate(kfold_options):

            if(not options.randsort):
                k_options.label = options.label + "_j1_kfold%i" % k
                k_options.output = options.output + "j1_kfold%i/" % k
                k_options.training_j = 1
                create_model_ensemble(k_options)

                k_options.label = options.label + "_j2_kfold%i" % k
                k_options.output = options.output + "j2_kfold%i/" % k
                k_options.training_j = 2
                create_model_ensemble(k_options)
            else:
                k_options.label = options.label + "_jrand_kfold%i" % k
                k_options.output = options.output + "jrand_kfold%i/" % k
                k_options.training_j = 1
                if(len(options.ae_dir) > 0):
                    if(options.mbin < 0):
                        print("Must supply mbin with ae_dir !")
                        exit(1)
                    else:
                        ae_mbin = k_options.mbin
                        if(ae_mbin == 6 or ae_mbin == 16 and not options.data): ae_mbin -= 1
                        k_options.labeler_name = options.ae_dir + "jrand_AE_kfold%i_mbin%i.h5" % (k, ae_mbin)
                create_model_ensemble(k_options)




    #get all the condor models
    if(get_condor and options.condor):
        for k,k_options in enumerate(kfold_options):

                c_opts = condor_options().parse_args([])
                c_opts.getEOS = True

                if(not options.randsort):
                    #c_opts.name = "j1_kfold%i" % k
                    c_opts.name = options.label + "_j1_kfold%i" % k
                    c_opts.outdir = options.output + "j1_kfold%i/" % k
                    if( not os.path.exists(c_opts.outdir)): os.system("mkdir %s" % c_opts.outdir)
                    doCondor(c_opts)
                    #c_opts.name = "j2_kfold%i" % k
                    c_opts.name = options.label + "_j2_kfold%i" % k
                    c_opts.outdir = options.output + "j2_kfold%i/" % k
                    if( not os.path.exists(c_opts.outdir)): os.system("mkdir %s" % c_opts.outdir)
                    doCondor(c_opts)
                else:
                    c_opts.name = options.label + "_jrand_kfold%i" % k
                    c_opts.outdir = options.output + "jrand_kfold%i/" % k
                    if( not os.path.exists(c_opts.outdir)): os.system("mkdir %s" % c_opts.outdir)
                    doCondor(c_opts)

                

    #select events
    if(do_selection):
        print(options.__dict__)
        for k,k_options in enumerate(kfold_options):
            selection_options = copy.deepcopy(k_options)
            selection_options.data_batch_list = k_options.holdouts
            selection_options.val_batch_list = None
            selection_options.num_models = options.lfolds
            selection_options.max_events = -1
            selection_options.val_max_events = -1

            if(not options.randsort): 
                selection_options.labeler_name = k_options.output + "{j_label}_kfold%i/" % k
                check_all_models_there(selection_options.labeler_name.format(j_label = "j1"), selection_options.num_models)
                check_all_models_there(selection_options.labeler_name.format(j_label = "j2"), selection_options.num_models)
            else: 
                selection_options.labeler_name = k_options.output + "jrand_kfold%i/" % k
                check_all_models_there(selection_options.labeler_name, selection_options.num_models)

            selection_options.output = k_options.output + "fit_inputs_kfold%i_eff{eff}.h5" % k
            selection_options.do_roc = True

            if((not options.condor)): #run selection locally
                selection(selection_options)

            else: #submit to condor
                selection_options.output = "fit_inputs_kfold%i_eff{eff}.h5" % k
                selection_options.local_storage = True
                selection_options.save_mem = True
                selection_options.fin = "../data/BB/"
                selection_options_dict = selection_options.__dict__
                selection_opts_fname  = options.output + "select_opts_%i.json" % k
                write_options_to_json(selection_options_dict,  selection_opts_fname)


                condor_dir = options.output + "selection_condor_jobs/"
                if( not os.path.exists(condor_dir)): os.system("mkdir %s" % condor_dir)
                c_opts = condor_options().parse_args([])
                c_opts.nJobs = 1
                c_opts.outdir = condor_dir

                base_script = "../condor/scripts/select_from_json.sh"
                select_script = condor_dir + "select%i_script.sh" % k 

                if(options.fin[-1] == "/"):
                    bb_name = options.fin.split("/")[-2]
                else:
                    bb_name = options.fin.split("/")[-1]

                os.system("cp %s %s" % (base_script, select_script))
                os.system("sed -i s/BB_NAME/%s/g %s" % (bb_name, select_script))
                os.system("sed -i s/BSTART/%i/g %s" % (selection_options.data_batch_list[0], select_script))
                os.system("sed -i s/BSTOP/%i/g %s" % (selection_options.data_batch_list[-1] + 1, select_script))
                if(len(options.sig_file) > 0):
                    sig_fn = options.sig_file.split("/")[-1]
                    os.system("sed -i 's/SIGFILE/%s/g' %s" % ("--sig_file " + sig_fn, select_script))
                else:
                    os.system("sed -i 's/SIGFILE//g' %s" % (select_script))
                f_select_script = open(select_script, "a")

                for eff in options.effs:
                    f_select_script.write("xrdcp -f %s ${1}" % selection_options.output.format(eff=eff))

                f_select_script.write("xrdcp -f %s ${1}" % selection_options.output.format(eff=options.effs[0]).replace(".h5", "_eff.npz"))
                f_select_script.write("xrdcp -f %s ${1}" % selection_options.output.format(eff=options.effs[0]).replace("fit_inputs", "sig_shape"))


                os.system("sed -i s/KFOLDNUM/%i/g %s" % (k, select_script))
                os.system("sed -i s/FNAME/fit_inputs*/g %s" % (select_script))

                c_opts.script = select_script

                c_opts.name = options.label + "_select_kfold%i" % k 
                c_opts.overwrite = True
                c_opts.sub = True
                #c_opts.sub = False

                #tar models together
                tarname = options.output + "models.tar"
                os.system("tar -cf %s %s --exclude=*/condor_jobs/*" %(tarname, options.output + "j*/"))

                inputs_list = [selection_opts_fname, tarname]
                c_opts.input = inputs_list
                if(options.condor_mem > 0):
                    c_opts.mem = options.condor_mem


                doCondor(c_opts)

    if(do_clean):
        condor_dir = options.output + "selection_condor_jobs/"
        os.system("rm %s/models.tar" % options.output)

        for k,k_options in enumerate(kfold_options):

            c_opts_name = options.label + "_select_kfold%i" % k 
            cmd = "rm %s/models.tar" % (condor_dir + c_opts_name)
            os.system(cmd)

            if(not options.randsort):
                output1 = options.output + "j1_kfold%i/" % k
                output2 = options.output + "j2_kfold%i/" % k
                os.system("rm %s/model*.h5" % output1)
                os.system("rm %s/model*.h5" % output2)

            else:
                output1 = options.output + "jrand_kfold%i/" % k
                os.system("rm %s/model*.h5" % output1)


    if(do_interp):
        for k,k_options in enumerate(kfold_options):
            interp_options = copy.deepcopy(k_options)
            interp_options.data_batch_list = k_options.holdouts
            interp_options.val_batch_list = None
            interp_options.num_models = options.lfolds
            #interp_options.num_models = 1


            if(not options.randsort): interp_options.labeler_name = k_options.output + "{j_label}_kfold%i/" % k
            else: interp_options.labeler_name = k_options.output + "jrand_kfold%i/" % k
            interp_options.output = interp_options.output + "interp/"
            os.system("mkdir " + interp_options.output)
            model_interp(interp_options)
            break


    if(do_merge):

        if(len(options.effs) == 0): options.effs = [mass_bin_select_effs[options.mbin] ]

        if(options.condor):
            for k,k_options in enumerate(kfold_options):

                c_opts = condor_options().parse_args([])
                c_opts.getEOS = True
                c_opts.outdir = options.output
                c_opts.name = options.label + "_select_kfold%i" % k 
                doCondor(c_opts)
                if(not os.path.exists(options.output + 'fit_inputs_kfold{k}.h5'.format(k=k))):
                    c_opts.name += 'x'
                #    doCondor(c_opts)

        #merge selections
        for eff in options.effs:
            fit_inputs_merge = options.output + "fit_inputs_eff{eff}.h5".format(eff = eff)

            #print(options.__dict__)
            if('eff_only' in options.__dict__.keys() and not options.eff_only): #merge fit inputs
                merge_cmd = "python ../../CASEUtils/H5_maker/H5_merge.py %s "  % fit_inputs_merge

                do_merge = False
                for k,k_options in enumerate(kfold_options):
                    fit_inputs = options.output + "fit_inputs_kfold{k}_eff{eff}.h5".format(k = k, eff = eff)

                    if(not os.path.exists(fit_inputs)):
                        fit_inputs = options.output + "fit_inputs_kfold{k}.h5".format(k = k)
                    if(os.path.exists(fit_inputs)): 
                        do_merge = True
                        merge_cmd += fit_inputs + " " 

                if(do_merge):
                    print("Merge cmd: " + merge_cmd)
                    subprocess.call(merge_cmd ,shell = True)
                else:
                    print("Unable to find files for merge")

            #avg efficiencies
            avg_inputs = []
            for k,k_options in enumerate(kfold_options):
                fit_inputs = options.output + "fit_inputs_kfold{k}_eff{eff}.h5".format(k = k, eff = eff)

                if(not os.path.exists(fit_inputs)):
                    fit_inputs = options.output + "fit_inputs_kfold{k}.h5".format(k = k)

                if(os.path.exists(fit_inputs)): avg_inputs.append(fit_inputs)

            avg_eff(fit_inputs_merge, avg_inputs)

            #merge signal shapes
            if(len(options.sig_file)):
                sig_shape_merge = options.output + "sig_shape_eff{eff}.h5".format(eff = eff)

                #print(options.__dict__)
                if('eff_only' in options.__dict__.keys() and not options.eff_only): #merge fit inputs
                    merge_cmd = "python ../../CASEUtils/H5_maker/H5_merge.py %s "  % sig_shape_merge

                    do_merge = False
                    for k,k_options in enumerate(kfold_options):
                        sig_shape = options.output + "sig_shape_kfold{k}_eff{eff}.h5".format(k = k, eff = eff)

                        if(not os.path.exists(sig_shape)):
                            sig_shape = options.output + "sig_shape_kfold{k}.h5".format(k = k)
                        if(os.path.exists(sig_shape)): 
                            do_merge = True
                            merge_cmd += sig_shape + " " 

                    if(do_merge):
                        print("Merge cmd: " + merge_cmd)
                        subprocess.call(merge_cmd ,shell = True)
                    else:
                        print("Unable to find files for signal shape merge")
                


    if(do_fit):

        fit_start = -1.
        if('fit_start' in options.__dict__.keys() and options.fit_start > 0):
            fit_start = options.fit_start
        run_fit = True

        if(len(options.effs) == 0):
            options.effs = [mass_bin_select_effs[options.mbin] ]

        if(len(options.sig_shape) > 0):
            sig_shape_file = base_path + options.sig_shape
        elif( 'mjj_sig' in options.__dict__.keys() and options.mjj_sig > 0):
            sig_shape_file = base_path + "../fitting/interpolated_signal_shapes/case_interpolation_M%.1f.root" % options.mjj_sig
        elif(not options.generic_sig_shape and os.path.exists(options.output + "sig_shape_eff{eff}.h5".format(eff = options.effs[0]))):
            #fit the signal shape
            sig_shape_h5 = base_path + options.output + 'sig_shape_eff{eff}.h5'.format(eff = options.effs[0])
            sig_fit_cmd = "python fit_signalshapes.py -i %s -o %s -M %i --dcb-model --fitRange 0.3 >& %s" % (sig_shape_h5, 
                                   base_path + options.output, options.mjj_sig, base_path + options.output + "sig_fit_log.txt")
            print(sig_fit_cmd)
            full_sig_fit_cmd = fit_cmd_setup + sig_fit_cmd + "; "  + fit_cmd_after

            subprocess.call(full_sig_fit_cmd,  shell = True, executable = '/bin/bash')
            sig_shape_file = base_path + options.output + 'sig_fit_%i.root' % options.mjj_sig
        else: sig_shape_file = ""


        final_fit_start = run_dijetfit(options, fit_start = fit_start, sig_shape_file = sig_shape_file, label = options.fit_label, loop = True)
        output = final_fit_start


    if(do_bias_test):

        fit_file = options.output + 'fit_results_%.1f.json' % options.mjj_sig
        if(not os.path.exists(fit_file)):
            print("Fit results file not found. Run regular fit before bias test!")
            exit(1)
        with open(fit_file, 'r') as f:
            fit_params = json.load(f, encoding="latin-1")
            exp_lim = fit_params['exp_lim_events']

        sig_shape_file = base_path + "../fitting/interpolated_signal_shapes/case_interpolation_M%.1f.root" % options.mjj_sig
        base_path = os.path.abspath(".") + "/"

        fit_inputs = 'fit_inputs_eff{eff}.h5'.format(eff = options.effs[0])
        input_file = base_path + options.output + fit_inputs
        alt_shape_ver = 4
        num_samples = 200

        outdir = options.output + "bias_test_alt%i/" % alt_shape_ver
        os.system("mkdir " + outdir)

        dijet_cmd = " python bias_test.py -i %s -p %s --rebin --alt_shape_ver %i --num_samples %i " % (input_file, base_path + outdir,  alt_shape_ver, num_samples)

        #roughly inject 0, 2sigma and 5sigma
        sigs = [0., 1.5 * exp_lim, 4.0 * exp_lim]
        labels = ["%isigma" % (sig) for sig in [0, 2, 5]]

        #sigs = [1.5 * exp_lim]
        #labels = ["2sigma"]


        for i,num_sig in enumerate(sigs):
            dijet_cmd_iter =  dijet_cmd + " -M %.0f --sig_shape %s --dcb-model -l %s --num_sig %i " % (options.mjj_sig, sig_shape_file,  labels[i], num_sig)
            dijet_cmd_iter += " >& %s/bias_test_log%.0f_nsig%i.txt " % (base_path + outdir, options.mjj_sig, num_sig)
            full_fit_cmd = fit_cmd_setup +  dijet_cmd_iter + "; " + fit_cmd_after
            print(full_fit_cmd)
            subprocess.call(full_fit_cmd,  shell = True, executable = '/bin/bash')




    stop_time = time.time()
    print("Total time taken was %s" % ( stop_time - start_time))


    #merge different selections

    if(do_roc):
        sig_effs = []
        bkg_effs = []
        labels = [ "kfold %i" % k for  k in range(options.kfolds)]
        colors = ["g", "b", "r", "gray", "purple", "pink", "orange", "m", "skyblue", "yellow"]
        for k,k_options in enumerate(kfold_options):
            np_fname = options.output + "fit_inputs_kfold{k}_eff{eff}_effs.npz".format(k=k, eff = options.effs[0])
            np_file = np.load(np_fname)
            sig_eff = np.clip(np_file["sig_eff"], 1e-6, 1.)
            bkg_eff = np.clip(np_file["bkg_eff"], 1e-6, 1.)

            bkg_eff_base = np.logspace(-4., 0., num = 1000)
            roc_label = ""
            if(options.redo_roc):
                print("Recomputing roc")
                j1_qs = np_file['j1_quantiles']
                j2_qs = np_file['j2_quantiles']
                Y = np_file['Y']
                sig_eff = []
                bkg_eff = []
                n_points = 200.
                roc_label = "_max"
                qcd_only = Y > -0.1
                #scores = np.minimum(j1_qs,j2_qs)
                #scores = np.maximum(j1_qs,j2_qs)
                scores = combine_scores(j1_qs, j2_qs, options.score_comb)
                bkg_eff, sig_eff, thresholds = roc_curve(Y[qcd_only], scores[qcd_only], drop_intermediate = True)


                sig_eff = np.clip(sig_eff, 1e-6, 1.)
                bkg_eff = np.clip(bkg_eff, 1e-6, 1.)


            
            #interpolate signal effs to smooth & standardize kfolds
            sig_eff_smooth = np.interp(bkg_eff_base, bkg_eff, sig_eff)
            #sig_eff_smooth = sig_eff
            
            sig_effs.append(sig_eff_smooth)
            #bkg_effs.append(bkg_eff)

        sig_effs_avg = np.mean(sig_effs, axis = 0)
        bkg_effs_avg = np.mean(sig_effs, axis = 0)
        labels.append("Avg.")
        sig_effs.append(sig_effs_avg)
        bkg_effs = [bkg_eff_base] * len(sig_effs)

        sic_fname = options.output + options.label + roc_label + "_sic.png"
        roc_fname = options.output + options.label + roc_label + "_roc.png"
        make_sic_plot(sig_effs, bkg_effs, colors = colors, labels = labels, fname = sic_fname)
        make_roc_plot(sig_effs, bkg_effs, colors = colors, labels = labels, fname = roc_fname)
        f_np = options.output + options.label + "_avg_tagging_effs.npz"
        print("Saving avg effs in  %s" % f_np)
        np.savez(f_np, sig_eff = sig_effs_avg, bkg_eff = bkg_eff_base)

    return output


if(__name__ == "__main__"):
    parser = input_options()
    parser.add_argument("--opts", default = "", help = "Options in json")
    parser.add_argument("--sig_norm_unc", default = -1.0, type = float, help = "parameter for fit (uncertainty on signal efficiency)")
    parser.add_argument("--ae_dir", default = "", help = "directory with all the autoencoders (auto pick the right mbin and kfold)")
    parser.add_argument("--effs", nargs="+", default = [], type = float)
    parser.add_argument("--kfolds", default = 5, type = int)
    parser.add_argument("--lfolds", default = 4, type = int)
    parser.add_argument("--numBatches", default = 40, type = int)
    parser.add_argument("--do_TNT",  default=False, action = 'store_true',  help="Use TNT (default cwola)")
    parser.add_argument("--do_ttbar",  default=False, action = 'store_true',  help="Do ttbar CR training")
    parser.add_argument("--condor", dest = 'condor', action = 'store_true')
    parser.add_argument("--no-condor", dest = 'condor', action = 'store_false')
    parser.add_argument("--fit_label", dest = 'fit_label', default = "")
    parser.add_argument("--sig_shape", default = "", help='signal shape file')
    parser.add_argument("--generic_sig_shape", default = False, action ='store_true')
    parser.add_argument("--no_generic_sig_shape", dest = 'generic_sig_shape', action ='store_false')
    parser.set_defaults(condor=False)
    parser.add_argument("--step", default = "train",  help = 'Which step to perform (train, get, select, fit, roc, all)')
    parser.add_argument("--fit_start", default = -1.,  type = float, help = 'Lowest mjj value for dijet fit')
    parser.add_argument("--redo_roc", default = False,  action = 'store_true', help = 'Remake roc')
    parser.add_argument("--reload", action = 'store_true', help = "Reload based on previously saved options")
    parser.add_argument("--condor_mem", default = -1, type = int, help = "Memory for condor jobs")
    parser.add_argument("--recover", dest='recover', action = 'store_true', help = "Retrain jobs that failed")
    parser.add_argument("--new", dest='reload', action = 'store_false', help = "Reload based on previously saved options")
    parser.set_defaults(reload=True)
    options = parser.parse_args()
    full_run(options)
