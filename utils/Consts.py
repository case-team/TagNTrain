
mass_bins1 = [1350., 1650., 2017., 2465., 3013., 3682., 4500., 5500., 6723., 99999.]
mass_bins2 = [1492., 1824., 2230., 2725., 3331., 4071., 4975., 6081., 7432., 99999.]

#From the h5 maker, maps between different systematics and their index
sys_weights_map = {
        'nom_weight' : 0,
        'pdf_up' : 1,
        'pdf_down': 2,
        'prefire_up': 3,
        'prefire_down' : 4,
        'pileup_up' : 5 ,
        'pileup_down' : 6,
        'btag_up' : 7,
        'btag_down' : 8,
        'PS_ISR_up' : 9,
        'PS_ISR_down' : 10,
        'PS_FSR_up' : 11,
        'PS_FSR_down' : 12,
        'F_up' : 13,
        'F_down' : 14,
        'R_up' : 15,
        'R_down' : 16,
        'RF_up' : 17,
        'RF_down' : 18
        }

JME_vars = { 'JES_up', 'JES_down', 'JER_up',   'JER_down', 'JMS_up',  'JMS_down', 'JMR_up',  'JMR_down' }

JME_vars_map = {
        'pt_JES_up' : 0,
        'm_JES_up' : 1,
        'pt_JES_down' : 2,
        'm_JES_down' : 3,
        'pt_JER_up' : 4,
        'm_JER_up' : 5,
        'pt_JER_down' : 6,
        'm_JER_down' : 7,
        'm_JMS_up' : 8,
        'm_JMS_down' : 9,
        'm_JMR_up' : 10,
        'm_JMR_down' : 11
        }

sys_list = sorted((set(sys_weights_map.keys()) | JME_vars))
sys_list.remove("nom_weight")
sys_weight_list = sorted(set(sys_weights_map.keys()))
sys_weight_list.remove("nom_weight")

sys_list_clean = { sys.replace("_up", "").replace("_down", "") for sys in sys_list}
sys_weight_list_clean = { sys.replace("_up", "").replace("_down", "") for sys in sys_weight_list}

JME_vars_clean = { sys.replace("_up", "").replace("_down", "") for sys in JME_vars}

