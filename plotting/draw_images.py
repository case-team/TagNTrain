import sys
sys.path.append('..')
from utils.TrainingUtils import *
import h5py


fin = "../data/BB_images_v1.h5"
#f_sig = "../data/BulkGravToZZToZhadZhad_narrow_M-2500.h5" #1
#f_sig = "../data/WprimeToWZToWhadZhad_narrow_M-3500.h5" #2
f_sig = "../data/WkkToWRadionToWWW_M2500-R0-08.h5" #3
f_bkg = "../data/QCD_only.h5"
sig_idx = 3
n_imgs = 10
n_mean_img = 20000

m_low = 2250.
m_high = 2750.
#m_low = 3150.
#m_high = 3850.

single_file = False

plot_dir = "../plots/BB1_test_june/images/"
model_dir = "../models/BB1_test_june/"

#model type: 0 is CNN, 1 is autoencoder, 2 is dense network



keys = ['mjj', 'j1_images', 'j2_images', 'jet_kinematics']

if(single_file):
    num_data = 300000
    data_start = 0
    data = DataReader(fin, signal_idx = sig_idx, start = data_start, stop = data_start + num_data, keys = keys, m_low = m_low, m_high = m_high )
    data.read()
    j1_images = data['j1_images']
    j2_images = data['j2_images']
    jet_kin = data['jet_kinematics']
    Y = data['label']
    bkg_events = Y < 0.1
    sig_events = Y > 0.9
    j1_im_sig = j1_images[sig_events]
    j2_im_sig = j2_images[sig_events]
    j1_im_bkg = j1_images[bkg_events]
    j2_im_bkg = j2_images[bkg_events]

    jet_kin_sig = jet_kin[sig_events]
    jet_kin_bkg = jet_kin[bkg_events]

else:
    bkg_start = 0
    n_bkg = 300000
    sig_start = 0
    sig_stop = -1
    d_bkg = DataReader(f_bkg, keys = keys, signal_idx = -1, start = bkg_start, stop = bkg_start + n_bkg, m_low = m_low, m_high = m_high )
    d_bkg.read()
    d_sig = DataReader(f_sig, keys = keys, signal_idx = -1, start = sig_start, stop = sig_stop, m_low = m_low, m_high = m_high )
    d_sig.read()

    j1_im_sig = d_sig['j1_images']
    j1_im_bkg = d_bkg['j1_images']
    j2_im_sig = d_sig['j2_images']
    j2_im_bkg = d_bkg['j2_images']

    jet_kin_sig = d_sig['jet_kinematics']
    jet_kin_bkg = d_bkg['jet_kinematics']


for i in range(n_imgs):
    img = j1_im_sig[i]
    pt = jet_kin_sig[i,2]
    m = jet_kin_sig[i,5]
    mjj = jet_kin_sig[i,0]
    title = "Signal Event, Heavy Jet: Mjj %.0f M = %.0f pT %.0f" % (mjj, m, pt)
    draw_jet_image(img, title, fname = plot_dir + "j1_s%i_%i.png" % (sig_idx, i))

    img = j2_im_sig[i]
    pt = jet_kin_sig[i,6]
    m = jet_kin_sig[i,9]
    mjj = jet_kin_sig[i,0]
    title = "Signal Event, Light Jet: Mjj %.0f M = %.0f pT %.0f" % (mjj, m, pt)
    draw_jet_image(img, title, fname = plot_dir + "j2_s%i_%i.png" % (sig_idx, i))


    img = j1_im_bkg[i]
    pt = jet_kin_bkg[i,2]
    m = jet_kin_bkg[i,5]
    mjj = jet_kin_bkg[i,0]
    title = "Background Event, Heavy Jet: Mjj %.0f M = %.0f pT %.0f" % (mjj, m, pt)
    draw_jet_image(img, title, fname = plot_dir + "j1_bkg_%i.png" % i)

    img = j2_im_bkg[i]
    pt = jet_kin_bkg[i,6]
    m = jet_kin_bkg[i,9]
    mjj = jet_kin_bkg[i,0]
    title = "Background Event, Light Jet: Mjj %.0f M = %.0f pT %.0f" % (mjj, m, pt)
    draw_jet_image(img, title, fname = plot_dir + "j2_bkg_%i.png" % i)

j1_mean_img_sig = np.mean(j1_im_sig[:n_mean_img], axis = 0)
title = "Signal Event, Heavy Jet Average Image"
draw_jet_image(j1_mean_img_sig, title, fname = plot_dir + "j1_s%i_avg.png" % sig_idx)

j2_mean_img_sig = np.mean(j2_im_sig[:n_mean_img], axis = 0)
title = "Signal Event, Light Jet Average Image" 
draw_jet_image(j2_mean_img_sig, title, fname = plot_dir + "j2_s%i_avg.png" % sig_idx)


j1_mean_img_bkg = np.mean(j1_im_bkg[:n_mean_img], axis =0)
title = "Background Event, Heavy Jet Average Image" 
draw_jet_image(j1_mean_img_bkg, title, fname = plot_dir + "j1_bkg_avg.png")

j2_mean_img_bkg = np.mean(j2_im_bkg[:n_mean_img], axis = 0)
title = "Background Event, Light Jet Average Image" 
draw_jet_image(j2_mean_img_bkg, title, fname = plot_dir + "j2_bkg_avg.png")
