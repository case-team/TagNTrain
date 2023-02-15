import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import gridspec
from sklearn.metrics import roc_curve,auc
from .Consts import *
import scipy.stats
import numpy as np

fig_size = (12,9)

def draw_mbins(plot, ymin = 0., ymax = 0., colors = ('blue', 'green')):
    for mass in mass_bins1:
        plot.axvline(mass, color = colors[0], linestyle = 'dashed', linewidth = 0.7)
    for mass in mass_bins2:
        plot.axvline(mass, color = colors[1], linestyle = 'dashed', linewidth = 0.7)

def add_patch(legend, patch, name):
    from matplotlib.patches import Patch
    ax = legend.axes

    handles, labels = ax.get_legend_handles_labels()
    handles.append(patch)
    labels.append(name)

    legend._legend_box = None
    legend._init_legend_box(handles, labels)
    legend._set_loc(legend._loc)
    legend.set_title(legend.get_title().get_text())




def print_image(a):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            print("%5.3e" % a[i,j], end = ' ')
        print("\n", end='')
    print("\n")

def plot_training(hist, fname =""):
    #plot trianing and validation loss

    loss = hist['loss']

    epochs = range(1, len(loss) + 1)
    colors = ['b', 'g', 'grey', 'r']
    idx=0

    plt.figure(figsize=fig_size)
    for label,loss_hist in hist.items():
        if(len(loss_hist) > 0): plt.plot(epochs, loss_hist, colors[idx], label=label)
        idx +=1
    plt.title('Training and validation loss')
    plt.yscale("log")
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    if(fname != ""): 
        plt.savefig(fname)
        print("saving fig %s" %fname)
    #else: 
        #plt.show(block=False)


def make_roc_curve(classifiers, y_true, colors = None, logy=True, labels = None, fname=""):
    plt.figure(figsize=fig_size)

    y_true = np.clip(y_true, 0,1)

    fs = 18
    fs_leg = 16
    sig_effs = []
    bkg_effs = []
    aucs = []
    for idx,scores in enumerate(classifiers):

        fpr, tpr, thresholds = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        sig_effs.append(tpr)
        bkg_effs.append(fpr)

    make_roc_plot(sig_effs, bkg_effs, colors = colors, labels = labels, logy = logy, fname = fname)
        
def make_roc_plot(sig_effs, bkg_effs, colors = None, logy = True, labels = None, fname = ""):

    plt.figure(figsize=fig_size)
    fs = 18
    fs_leg = 16
    sic_max = 10.

    for idx in range(len(sig_effs)):
        tpr = sig_effs[idx]
        fpr = bkg_effs[idx]
        auc_ = auc(fpr, tpr)

        fpr = np.clip(fpr, 1e-8, 1.)
        #guard against division by 0
        ys = 1./fpr

        lbl = 'auc %.3f' % auc_
        clr = 'navy'
        if(labels is not None): lbl = labels[idx] + " = %.3f" % auc_
        if(colors is not None): clr = colors[idx]

        print(lbl)

        plt.plot(tpr, ys, lw=2, color=clr, label=lbl)



    plt.xlim([0, 1.0])
    plt.xlabel('Signal Efficiency', fontsize=fs)
    if(logy): 
        plt.yscale('log')
    plt.ylim([1., 1e4])
    plt.ylabel('QCD Rejection Rate', fontsize=fs)

    plt.legend(loc="upper right", fontsize = fs_leg)
    if(fname != ""): 
        print("Saving roc plot to %s" % fname)
        plt.savefig(fname)
    #else: 
        #plt.show(block=False)


def make_sic_plot(sig_effs, bkg_effs, colors = None, labels = None, eff_min = 1e-3, ymax = -1, fname=""):


    plt.figure(figsize=fig_size)

    fs = 18
    fs_leg = 16
    sic_max = 10.



    for idx in range(len(sig_effs)):
        sic = sig_effs[idx] / np.sqrt(bkg_effs[idx])

        lbl = 'auc'
        clr = 'navy'
        if(labels != None): lbl = labels[idx]
        if(colors != None): clr = colors[idx]
        print(lbl, "max sic: ", np.amax(sic))
        plt.plot(bkg_effs[idx], sic, lw=2, color=clr, label='%s' % lbl)


    
    plt.xlim([eff_min, 1.0])
    if(ymax < 0):
        ymax = sic_max
        
    plt.ylim([0,ymax])
    plt.xscale('log')
    plt.xlabel('Background Efficiency', fontsize = fs)
    plt.ylabel('Significance Improvement', fontsize = fs)
    plt.tick_params(axis='x', labelsize=fs_leg)
    plt.tick_params(axis='y', labelsize=fs_leg)
    plt.grid(axis = 'y', linestyle='--', linewidth = 0.5)
    plt.legend(loc="best", fontsize= fs_leg)
    if(fname != ""):
        plt.savefig(fname)
        print("Saving file to %s " % fname)


def make_sic_curve(classifiers, y_true, colors = None, logy=False, labels = None, eff_min = 1e-3, ymax = -1, fname=""):

    y_true = np.clip(y_true, 0,1)
    plt.figure(figsize=fig_size)

    fs = 18
    fs_leg = 16

    sic_max = 10.



    for idx,scores in enumerate(classifiers):
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        fpr= np.clip(fpr, 1e-8, 1.)
        tpr = np.clip(tpr, 1e-8, 1.)

        mask = tpr > eff_min

        xs = fpr[mask]
        ys = tpr[mask]/np.sqrt(xs)

        sic_max = max(np.amax(ys), sic_max)


        


        lbl = 'auc'
        clr = 'navy'
        if(labels != None): lbl = labels[idx]
        if(colors != None): clr = colors[idx]
        print(lbl, "max sic: ", np.amax(ys))
        plt.plot(xs, ys, lw=2, color=clr, label='%s' % lbl)


    
    plt.xlim([eff_min, 1.0])
    if(ymax < 0):
        ymax = sic_max
        
    plt.ylim([0,ymax])
    plt.xscale('log')
    plt.xlabel('Background Efficiency', fontsize = fs)
    plt.ylabel('Significance Improvement', fontsize = fs)
    plt.tick_params(axis='x', labelsize=fs_leg)
    plt.tick_params(axis='y', labelsize=fs_leg)
    plt.grid(axis = 'y', linestyle='--', linewidth = 0.5)
    plt.legend(loc="best", fontsize= fs_leg)
    if(fname != ""):
        plt.savefig(fname)
        print("Saving file to %s " % fname)

def make_histogram(entries, labels, colors, xaxis_label="", title ="", num_bins = 10, logy = False, normalize = False, stacked = False, h_type = 'step', 
        h_range = None, fontsize = 16, fname="", yaxis_label = "", ymax = -1):
    alpha = 1.
    if(stacked): 
        h_type = 'barstacked'
        alpha = 0.2
    fig = plt.figure(figsize=fig_size)
    ns, bins, patches = plt.hist(entries, bins=num_bins, range=h_range, color=colors, alpha=alpha,label=labels, density = normalize, histtype=h_type)
    plt.xlabel(xaxis_label, fontsize =fontsize)
    plt.tick_params(axis='x', labelsize=fontsize)

    if(logy): plt.yscale('log')
    elif(ymax > 0):
        plt.ylim([0,ymax])
    else:
        ymax = 1.3 * np.amax(ns)
        plt.ylim([0,ymax])

    if(yaxis_label != ""):
        plt.ylabel(yaxis_label, fontsize=fontsize)
        plt.tick_params(axis='y', labelsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.legend(loc='upper right', fontsize = fontsize)
    if(fname != ""): 
        plt.savefig(fname)
        print("saving fig %s" %fname)
    #else: plt.show(block=False)
    return fig

def make_profile_hist(x,y, x_bins, xaxis_label="", yaxis_label="", fname = "", fontsize = 16, logy=False):

    x_bins_ids = np.digitize(x, bins = x_bins)
    bin_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
    bin_width = x_bins[1] - x_bins[0]
    y_means = scipy.stats.binned_statistic(x, y, bins = x_bins, statistic = "mean").statistic
    y_sem = scipy.stats.binned_statistic(x, y, bins = x_bins, statistic=scipy.stats.sem).statistic

    fig = plt.figure(figsize=fig_size)
    plt.errorbar(x=bin_centers, y=y_means, xerr=bin_width, yerr=y_sem, linestyle='none')

    plt.xlabel(xaxis_label, fontsize =fontsize)
    plt.tick_params(axis='x', labelsize=fontsize)
    if(logy): plt.yscale('log')
    if(yaxis_label != ""):
        plt.ylabel(yaxis_label, fontsize=fontsize)
        plt.tick_params(axis='y', labelsize=fontsize)
    if(fname != ""): 
        print("saving fig %s" %fname)
        plt.savefig(fname)
    #else: plt.show(block=False)
    return fig



def make_outline_hist(stacks,outlines, labels, colors, xaxis_label, title, num_bins, logy = False,  normalize = False, save=False, h_type = 'step', 
        h_range = None, fontsize = 16, fname="", yaxis_label = ""):
    alpha = 1.
    n_stacks = len(stacks)
    fig = plt.figure(figsize=fig_size)
    if(n_stacks > 0):
        plt.hist(stacks, bins=num_bins, range=h_range, color=colors[:n_stacks], alpha=0.2,label=labels[:n_stacks], density = normalize, histtype='barstacked')
    if(len(outlines) > 0):
        plt.hist(outlines, bins=num_bins, range=h_range, color=colors[n_stacks:], alpha=1.,label=labels[n_stacks:], density = normalize, histtype='step')
    plt.xlabel(xaxis_label, fontsize =fontsize)
    plt.tick_params(axis='x', labelsize=fontsize)
    if(logy): plt.yscale('log')
    if(yaxis_label != ""):
        plt.ylabel(yaxis_label, fontsize=fontsize)
        plt.tick_params(axis='y', labelsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.legend(loc='upper right', fontsize = fontsize)
    if(save): 
        print("saving fig %s" %fname)
        plt.savefig(fname)
    #else: plt.show(block=False)
    return fig



def make_multi_ratio_histogram(entries, labels, colors, axis_label, title, num_bins, normalize = False, save=False, h_range = None, 
        weights = None, fname="", ratio_range = -1, errors = False, logy = False, max_rw = 5, unc_band_norm = -1):
    h_type= 'step'
    alpha = 1.
    fontsize = 22
    fig = plt.figure(figsize = fig_size)
    gs = gridspec.GridSpec(2,1, height_ratios = [3,1])
    ax0 =  plt.subplot(gs[0])

    low = np.amin(entries[0])
    high = np.amax(entries[0])
    if(h_range == None):
        h_range = (low, high)

    ns, bins, patches  = ax0.hist(entries, bins=num_bins, range=h_range, color=colors, alpha=alpha,label=labels[:len(entries)], 
            density = normalize, weights = weights, histtype=h_type)


    plt.xlim([low, high])
    if(logy): plt.yscale("log")
    plt.title(title, fontsize=fontsize)

    bin_size = bins[1] - bins[0]
    bincenters = 0.5*(bins[1:]+bins[:-1]) 
    ax1 = plt.subplot(gs[1])

    ratios = []
    errs = []

    leg = ax0.legend(loc='best', fontsize = 14)
    


    frac_unc = None
    if(unc_band_norm > 0):
        raw_dist = np.copy(ns[0])
        if(not normalize): norm_dist = raw_dist * unc_band_norm / np.sum(raw_dist)
        else: normed_dist = raw_dist * bin_size * unc_band_norm
        #print(raw_dist)
        unc_dist = np.sqrt(normed_dist)
        frac_unc = unc_dist / normed_dist
        bincenters_band = np.copy(bincenters)
        bincenters_band[0] = low
        bincenters_band[-1] = high

        if(len(labels) <= len(entries)): label = "Signal Injection Stat. Unc."
        else: label = labels[-1]

        l_unc = ax1.fill_between(bincenters_band, 1. - frac_unc, 1. + frac_unc, color = 'gray', alpha = 0.5, label = labels)
        add_patch(leg, l_unc, label)

    for i in range(1, len(ns)):
        ratio = np.clip(ns[i], 1e-8, None) / np.clip(ns[0], 1e-8, None)
        ratios.append(ratio)

        ratio_err = None
        if(errors):
            if(weights != None):
                w0 = weights[0]**2
                w1 = weights[i]**2
                norm0 = np.sum(weights[0])*bin_size
                norm1 = np.sum(weights[i])*bin_size
            else:
                w0 = w1 = None
                norm0 = entries[0].shape[0]*bin_size
                norm1 = entries[i].shape[0]*bin_size

            err0 = np.sqrt(np.histogram(entries[0], bins=bins, weights=w0)[0])/norm0
            err1 = np.sqrt(np.histogram(entries[i], bins=bins, weights=w1)[0])/norm1
            err0_alt  = np.sqrt(norm0*n0)/norm0
            err_alt1  = np.sqrt(norm1*n1)/norm1
            ratio_err = ratio * np.sqrt((err0/n0)**2 + (err1/n1)**2)
        errs.append(ratio_err)

        ax1.errorbar(bincenters, ratio, yerr = ratio_err, alpha=alpha, markerfacecolor = colors[i], markeredgecolor = colors[i], fmt='ko')



    

    label_size = 18
    ax1.set_ylabel("Ratio", fontsize= label_size)
    ax1.set_xlabel(axis_label, fontsize = label_size)

    plt.xlim([low, high])

    if(type(ratio_range) == list):
        plt.ylim(ratio_range[0], ratio_range[1])
    else:
        if(ratio_range > 0):
            plt.ylim([1-ratio_range, 1+ratio_range])

    plt.grid(axis='y')

    if(save): 
        plt.savefig(fname)
        print("saving fig %s" %fname)

    return ns, bins, ratios, frac_unc



def make_ratio_histogram(entries, labels, colors, axis_label, title, num_bins, normalize = False, save=False, h_range = None, 
        weights = None, fname="", ratio_range = -1, errors = False, extras = None, logy = False, max_rw = 5):
    h_type= 'step'
    alpha = 1.
    fontsize = 24
    fig = plt.figure(figsize = fig_size)
    gs = gridspec.GridSpec(2,1, height_ratios = [3,1])
    ax0 =  plt.subplot(gs[0])

    low = np.amin([np.amin(entries[0]), np.amin(entries[1])])
    high = np.amax([np.amax(entries[0]), np.amax(entries[1])])
    if(h_range is None):
        h_range = (low, high)

    ns, bins, patches  = ax0.hist(entries, bins=num_bins, range=h_range, color=colors, alpha=alpha,label=labels, 
            density = normalize, weights = weights, histtype=h_type)

    if(extras is not None):
        ecolors = ['red', 'orange', 'cyan']
        for e_i, extra in enumerate(extras):
            ax0.hist(extra[0], bins= num_bins, range = h_range, color = ecolors[e_i], label = extra[2], density = normalize, weights = extra[1], histtype=h_type)


    plt.xlim([low, high])
    if(logy): plt.yscale("log")
    ax0.legend(loc='upper right', fontsize = fontsize * 0.75)
    plt.title(title, fontsize=fontsize)
    n0 = np.clip(ns[0], 1e-8, None)
    n1 = np.clip(ns[1], 1e-8, None)
    ratio =  n0/ n1

    #low outliers more ok than high ones
    if(max_rw > 0):
        ratio = np.clip(ratio, 1./(2*max_rw), max_rw)

    ratio_err = None

    bin_size = bins[1] - bins[0]

    if(errors):
        if(weights != None):
            w0 = weights[0]**2
            w1 = weights[1]**2
            norm0 = np.sum(weights[0])*bin_size
            norm1 = np.sum(weights[1])*bin_size
        else:
            w0 = w1 = None
            norm0 = entries[0].shape[0]*bin_size
            norm1 = entries[1].shape[0]*bin_size

        err0 = np.sqrt(np.histogram(entries[0], bins=bins, weights=w0)[0])/norm0
        err1 = np.sqrt(np.histogram(entries[1], bins=bins, weights=w1)[0])/norm1
        err0_alt  = np.sqrt(norm0*n0)/norm0
        err_alt1  = np.sqrt(norm1*n1)/norm1


        ratio_err = ratio * np.sqrt((err0/n0)**2 + (err1/n1)**2)

    bincenters = 0.5*(bins[1:]+bins[:-1]) 
    ax1 = plt.subplot(gs[1])

    ax1.errorbar(bincenters, ratio, yerr = ratio_err, alpha=alpha, fmt='ko')

    #plt.xlim([np.amin(entries[0]), np.amax(entries[0])])
    plt.xlim([low,high])
    ax1.set_ylabel("Ratio", fontsize = fontsize)
    ax1.set_xlabel(axis_label, fontsize = fontsize)


    if(type(ratio_range) == list):
        plt.ylim(ratio_range[0], ratio_range[1])
    else:
        if(ratio_range > 0):
            plt.ylim([1-ratio_range, 1+ratio_range])

    plt.grid(axis='y')

    if(save): 
        plt.savefig(fname)
        print("saving fig %s" %fname)

    return bins, ratio

def draw_jet_image(image, title, fname = "", do_log = False):
    fontsize = 20
    image = np.clip(np.squeeze(image).astype('float'), 1e-8, None)
    if(do_log): image = np.log(image)
    fig = plt.figure(figsize=fig_size)
    plt.imshow(image, cmap = 'Blues', interpolation = 'nearest')
    plt.title(title, fontsize = fontsize)
    if(fname != ""):
        plt.savefig(fname)


def make_graph(x, y, labels, colors, axis_names, fname= ""  ):

    fig, ax = plt.subplots()
    size = 20.
    x = np.array(x)
    if(type(colors) == list):
        for i in range(x.shape[0]):
            ax.scatter(x[i],y[i], c = colors[i], s=size, label = labels[i])
            
        plt.legend()
    else: ax.scatter(x,y, alpha = alpha, c = colors, s=size, label = labels)

    ax.set_xlabel(axis_names[0], fontsize=14)
    ax.set_ylabel(axis_names[1], fontsize=14)
    plt.tick_params(axis='y', labelsize=12)
    plt.tick_params(axis='x', labelsize=12)
    if(fname != ""):
        print("saving %s" % fname)
        plt.savefig(fname)


def make_scatter_plot(x, y, color, axis_names, fname= ""  ):

    fig, ax = plt.subplots()
    alpha = 0.5
    size = 0.4
    ax.scatter(x,y, alpha = alpha, c = color, s=size)

    correlation = np.corrcoef(x,y)[0,1]
    text_str = r'$\rho_{x,y} $ = %.3f' % correlation
    plt.annotate(text_str, xy = (0.05, 0.95), xycoords = 'axes fraction', fontsize=14)

    ax.set_xlabel(axis_names[0], fontsize=14)
    ax.set_ylabel(axis_names[1], fontsize=14)
    plt.tick_params(axis='y', labelsize=12)
    plt.tick_params(axis='x', labelsize=12)
    if(fname != ""):
        print("saving %s" % fname)
        plt.savefig(fname)

def horizontal_bar_chart(vals, labels, fname = "", xaxis_label = ""):
    fig, ax = plt.subplots()

    order = np.flip(vals.argsort())
    labels = np.array(labels)

    y_pos = np.arange(len(vals))
    ax.barh(y_pos, vals[order], align = 'center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels[order])
    ax.invert_yaxis()
    ax.set_xlabel(xaxis_label)

    if(fname != ""):
        print("saving %s" % fname)
        plt.savefig(fname)

