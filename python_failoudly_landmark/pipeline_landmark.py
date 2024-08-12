# get from https://github.com/steverab/failing-loudly
# may/might have been modified
# -------------------------------------------------
# IMPORTS
# sample size for multiv is changed[line 140s]
# the mpl.rParams is changed[line 40s]
# -------------------------------------------------

from urllib.request import pathname2url
import numpy as np
import seaborn as sns
import tensorflow as tf
seed = 1
# np.random.seed(seed)
# set_random_seed(seed)
tf.random.set_seed(seed)

import keras
import tempfile
import keras.models

from keras import backend as K 
from shift_detector import *
from shift_locator import *
from shift_applicator import *
from data_utils import *
from shared_utils import *
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc

# -------------------------------------------------
# PLOTTING HELPERS
# -------------------------------------------------


rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)
rc('axes', labelsize=22)
rc('xtick', labelsize=22)
rc('ytick', labelsize=22)
rc('legend', fontsize=13)

#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
mpl.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
#plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath}"


def clamp(val, minimum=0, maximum=255):
    if val < minimum:
        return minimum
    if val > maximum:
        return maximum
    return val


def colorscale(hexstr, scalefactor):
    hexstr = hexstr.strip('#')

    if scalefactor < 0 or len(hexstr) != 6:
        return hexstr

    r, g, b = int(hexstr[:2], 16), int(hexstr[2:4], 16), int(hexstr[4:], 16)

    r = clamp(r * scalefactor)
    g = clamp(g * scalefactor)
    b = clamp(b * scalefactor)

    return "#%02x%02x%02x" % (int(r), int(g), int(b))


def errorfill(x, y, yerr, color=None, alpha_fill=0.2, ax=None, fmt='-o', label=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = next(ax._get_lines.prop_cycler)['color']
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.semilogx(x, y, fmt, color=color, label=label)
    ax.fill_between(x, np.clip(ymax, 0, 1), np.clip(ymin, 0, 1), color=color, alpha=alpha_fill)


def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


linestyles = ['-', '-.', '--', ':']
brightness = [1.25, 1.0, 0.75, 0.5]
format = ['-o', '-h', '-p', '-s', '-D', '-<', '->', '-X']
markers = ['o', 'h', 'p', 's', 'D', '<', '>', 'X']
colors_old = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
colors = ['#2196f3', '#f44336', '#9c27b0', '#64dd17', '#009688', '#ff9800', '#795548', '#607d8b']
make_keras_picklable()
# -------------------------------------------------
# CONFIG
# -------------------------------------------------
# Significance level.
sign_level = 0.1
# Number of random runs to average results over.
random_runs = 1

# change line 121, 122, and [148 149] accordingly
hypothesis_test = "_LMSW"
# sys.argv = [ 'pipeline.py', 'mnist', 'small_image_shift', 'multiv']
datset = sys.argv[1]
shift_type = sys.argv[2]
test_type = sys.argv[3]


# Define results path and create directory.
path = './paper_results/'
path += test_type + '/'

#create path for csv tables
path1 = path
path1 += 'csv_tables' + '/'
if not os.path.exists(path1):
    os.makedirs(path1)

path += datset + '_'
path += sys.argv[2] + hypothesis_test + '/'
if not os.path.exists(path):
    os.makedirs(path)

# Define DR methods.
dr_techniques = [DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value, DimensionalityReduction.SRP.value, DimensionalityReduction.UAE.value, DimensionalityReduction.TAE.value, DimensionalityReduction.BBSDs.value, DimensionalityReduction.BBSDh.value]
#dr_techniques = [DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value, DimensionalityReduction.SRP.value, DimensionalityReduction.UAE.value, DimensionalityReduction.TAE.value, DimensionalityReduction.BBSDs.value]
if test_type == 'multiv':
    dr_techniques = [DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value, DimensionalityReduction.SRP.value, DimensionalityReduction.UAE.value, DimensionalityReduction.TAE.value, DimensionalityReduction.BBSDs.value]
if test_type == 'univ':
    dr_techniques_plot = dr_techniques.copy()
    dr_techniques_plot.append(DimensionalityReduction.Classif.value)
else:
    dr_techniques_plot = dr_techniques.copy()

# Define test types and general test sample sizes.
test_types = [td.value for td in TestDimensionality]
if test_type == 'multiv':
    od_tests = []
    if hypothesis_test == "_LMSW_":
        md_tests = [MultidimensionalTest.LMSW.value]
    elif hypothesis_test == "_MMD_":
        md_tests = [MultidimensionalTest.MMD.value]
    else:
        md_tests = [MultidimensionalTest.LMSW.value]

    #md_tests = [MultidimensionalTest.MMD.value]
    #samples = [10, 50, 100, 200, 500, 1000]
    samples = [1000]
else:
    od_tests = [OnedimensionalTest.LMSW.value]
    #od_tests = [OnedimensionalTest.KS.value]
    md_tests = []
    #samples = [10, 20, 50,  100, 200, 500, 1000, 10000]
    samples =  [10, 100, 200, 500, 1000, 5000]
difference_samples = 10

# Whether to calculate accuracy for malignancy quantification.
calc_acc = True

# Define shift types.
if sys.argv[2] == 'small_gn_shift':
    shifts = ['small_gn_shift_0.1',
              'small_gn_shift_0.5',
              'small_gn_shift_1.0']
elif sys.argv[2] == 'medium_gn_shift':
    shifts = ['medium_gn_shift_0.1',
              'medium_gn_shift_0.5',
              'medium_gn_shift_1.0']
elif sys.argv[2] == 'large_gn_shift':
    shifts = ['large_gn_shift_0.1',
              'large_gn_shift_0.5',
              'large_gn_shift_1.0']
elif sys.argv[2] == 'adversarial_shift':
    shifts = ['adversarial_shift_0.1',
              'adversarial_shift_0.5',
              'adversarial_shift_1.0']
elif sys.argv[2] == 'ko_shift':
    shifts = ['ko_shift_0.1',
              'ko_shift_0.5',
              'ko_shift_1.0']
    if test_type == 'univ':
        samples = [10, 20, 50, 100, 200, 500, 1000, 9000]
elif sys.argv[2] == 'orig':
    shifts = ['rand', 'orig']
    brightness = [1.25, 0.75]
elif sys.argv[2] == 'small_image_shift':
    shifts = ['small_img_shift_0.1',
              'small_img_shift_0.5',
              'small_img_shift_1.0']
elif sys.argv[2] == 'medium_image_shift':
    shifts = ['medium_img_shift_0.1',
              'medium_img_shift_0.5',
              'medium_img_shift_1.0']
elif sys.argv[2] == 'large_image_shift':
    shifts = ['large_img_shift_0.1',
              'large_img_shift_0.5',
              'large_img_shift_1.0']
elif sys.argv[2] == 'medium_img_shift+ko_shift':
    shifts = ['medium_img_shift_0.5+ko_shift_0.1',
              'medium_img_shift_0.5+ko_shift_0.5',
              'medium_img_shift_0.5+ko_shift_1.0']
    if test_type == 'univ':
        samples = [10, 20, 50, 100, 200, 500, 1000, 9000]
elif sys.argv[2] == 'only_zero_shift+medium_img_shift':
    shifts = ['only_zero_shift+medium_img_shift_0.1',
              'only_zero_shift+medium_img_shift_0.5',
              'only_zero_shift+medium_img_shift_1.0']
    samples = [10, 20, 50, 100, 200, 500, 1000]
else:
    shifts = []
    
if datset == 'coil100' and test_type == 'univ':
    samples = [10, 20, 50, 100, 200, 500, 1000, 2400]

if datset == 'mnist_usps':
    samples = [10, 20, 50, 100, 200, 500, 1000]

# -------------------------------------------------
# PIPELINE START
# -------------------------------------------------

# Stores p-values for all experiments of a shift class.
samples_shifts_rands_dr_tech = np.ones((len(samples), len(shifts), random_runs, len(dr_techniques_plot))) * (-1)
max_landmark_values = np.ones(( len(dr_techniques_plot),len(shifts))) * (-1)
max_landmark_p_values = np.ones(( len(dr_techniques_plot),len(shifts))) * (-1)

red_dim = -1
red_models = [None] * len(DimensionalityReduction)

# Iterate over all shifts in a shift class.
for shift_idx, shift in enumerate(shifts):

    # create folders for each shift
    shift_path = path + shift + '/'
    if not os.path.exists(shift_path):
        os.makedirs(shift_path)

    # Stores p-values for a single shift.
    rand_run_p_vals = np.ones((len(samples), len(dr_techniques_plot), random_runs)) * (-1)

    # Stores accuracy values for malignancy detection.
    val_accs = np.ones((random_runs, len(samples))) * (-1)
    te_accs = np.ones((random_runs, len(samples))) * (-1)
    dcl_accs = np.ones((len(samples), random_runs)) * (-1)

    #print(range(5, 10))

    # Average over a few random runs to quantify robustness.
    for rand_run in range(0, random_runs):

        #print("Random run %s/%s shift %s/3"  % (rand_run+1,random_runs,shift_idx+1))

        # create folders for each run
        rand_run_path = shift_path + str(rand_run) + '/'
        if not os.path.exists(rand_run_path):
            os.makedirs(rand_run_path)

        # np.random.seed(rand_run) 
        tf.random.set_seed(rand_run)

        # Load data.
        (X_tr_orig, y_tr_orig), (X_val_orig, y_val_orig), (X_te_orig, y_te_orig), orig_dims, nb_classes = \
            import_dataset(datset, shuffle=True)

        # normalize the data    
        X_tr_orig = normalize_datapoints(X_tr_orig, 255.)
        X_te_orig = normalize_datapoints(X_te_orig, 255.)
        X_val_orig = normalize_datapoints(X_val_orig, 255.)

        # Apply shift.
        if shift == 'orig':
            print('Original')
            (X_tr_orig, y_tr_orig), (X_val_orig, y_val_orig), (X_te_orig, y_te_orig), orig_dims, nb_classes = import_dataset(datset)
            X_tr_orig = normalize_datapoints(X_tr_orig, 255.)
            X_te_orig = normalize_datapoints(X_te_orig, 255.)
            X_val_orig = normalize_datapoints(X_val_orig, 255.)
            X_te_1 = X_te_orig.copy()
            y_te_1 = y_te_orig.copy()
        else:
            (X_te_1, y_te_1) = apply_shift(X_te_orig, y_te_orig, shift, orig_dims, datset)

        X_te_2 , y_te_2 = random_shuffle(X_te_1, y_te_1)

        # Check detection performance for different numbers of samples from test.
        for si, sample in enumerate(samples):

            #print("Sample size %s" % sample)
            #print("shift %s/%s RandomRun %s/%s SampleSize %s"  % (shift_idx+1,len(shifts),rand_run+1,random_runs,sample))
            print("%s: shift %s/%s(%s) RandomRun %s/%s SampleSize %s"  % (datset,shift_idx+1,len(shifts),sys.argv[2],rand_run+1,random_runs,sample))

            sample_path = rand_run_path + str(sample) + '/'
            if not os.path.exists(sample_path):
                os.makedirs(sample_path)

            X_te_3 = X_te_2[:sample,:]
            y_te_3 = y_te_2[:sample]

            if test_type == 'multiv':
                X_val_3 = X_val_orig[:1000,:]
                y_val_3 = y_val_orig[:1000]
            else:
                X_val_3 = X_val_orig[:sample,:]
                y_val_3 = y_val_orig[:sample]

            #X_tr_3 = np.copy(X_tr_orig)
            #y_tr_3 = np.copy(y_tr_orig)
            X_tr_3 = np.copy(X_tr_orig[:1000,:])
            y_tr_3 = np.copy(y_tr_orig[:1000])


            # print(nb_classes)

            # Detect shift.
            shift_detector = ShiftDetector(dr_techniques, test_types, od_tests, md_tests, sign_level, red_models,
                                           sample, datset)

            (od_decs, ind_od_decs, ind_od_p_vals), \
            (md_decs, ind_md_decs, ind_md_p_vals), \
            red_dim, red_models, val_acc, te_acc , (landmark_divs,V) = shift_detector.detect_data_shift(X_tr_3, y_tr_3, X_val_3, y_val_3,
                                                                                    X_te_3, y_te_3, orig_dims,
                                                                                    nb_classes)

            val_accs[rand_run, si] = val_acc
            te_accs[rand_run, si] = te_acc
            
            if test_type == 'multiv':
                print("Shift decision: ", ind_md_decs.flatten())
                print("Shift p-vals: ", ind_md_p_vals.flatten())

                rand_run_p_vals[si,:,rand_run] = ind_md_p_vals.flatten()
            else:
                print("Shift decision: ", ind_od_decs.flatten())
                print("Shift p-vals: ", ind_od_p_vals.flatten())
                 
                if DimensionalityReduction.Classif.value not in dr_techniques_plot:
                    rand_run_p_vals[si,:,rand_run] = ind_od_p_vals.flatten()
                    continue
########################################### suppose to be in one dimension below
            # Characterize shift via domain classifier.

            #shift_locator = ShiftLocator(orig_dims, dc=DifferenceClassifier.FFNNDCL, sign_level=sign_level)
            #model, score, (X_tr_dcl, y_tr_dcl, y_tr_old, X_te_dcl, y_te_dcl, y_te_old) = shift_locator.build_model(X_tr_3, y_tr_3, X_te_3, y_te_3)
            #test_indices, test_perc, dec, p_val = shift_locator.most_likely_shifted_samples(model, X_te_dcl, y_te_dcl)
            
            #shift_locator = ShiftLocator(orig_dims, dc=DifferenceClassifier.LMSW, sign_level=sign_level)
            #model, score, (X_tr_dcl, y_tr_dcl, y_tr_old, X_te_dcl, y_te_dcl, y_te_old) = shift_locator.build_model(X_tr_3, y_tr_3, X_te_3, y_te_3)
                          #x_tr_new, y_tr_new, y_tr_old, x_te_new, y_te_new, y_te_old
            #test_indices, test_perc, dec, p_val = shift_locator.most_likely_shifted_samples(model, X_te_dcl, y_te_dcl)
            #most_conf_test_indices, most_conf_test_perc, p_val < self.sign_level, p_val

            X_tr_dcl = X_tr_orig[:sample]
            y_tr_dcl = y_tr_orig[:sample]

            X_te_dcl = X_te_3
            y_te_dcl = y_te_3
            y_te_old = y_te_3
            #test_indices = np.argsort(y_te_new_pred[:,1])[::-1]
            #test_perc = np.sort(y_te_new_pred[:,1])[::-1]

            x_idx = np.concatenate((np.ones(X_te_3.shape[0]),np.zeros(X_te_3.shape[0])),axis=0).astype(bool) #bool indicator
            landmark_divs = landmark_divs.squeeze()
            #print(landmark_divs)
            V = V.squeeze()
            Z = np.concatenate((X_tr_dcl,X_te_dcl),axis=0)
            #np.set_printoptions(precision=4, suppress=True)

            Y = np.concatenate((y_tr_dcl,y_te_dcl),axis=0)
            
            #dr_check = 4 
            p_vals = ind_md_p_vals
            #p_val = p_vals[dr_check]

            landmark_sort_indeces = np.argsort(landmark_divs)[:,::-1]
            landmark_sort_values = np.sort(landmark_divs)[:,::-1]
            #dec = p_val < sign_level
            test_perc = landmark_sort_values
            test_indices = landmark_sort_indeces



            if datset == 'mnist':
                samp_shape = (28,28)
                cmap = 'gray'
            elif datset == 'cifar10' or datset == 'cifar10_1' or datset == 'coil100' or datset == 'svhn':
                samp_shape = (32,32,3)
                cmap = None
            elif datset == 'mnist_usps':
                samp_shape = (16,16)
                cmap = 'gray'
            elif datset == 'fashion_mnist':
                samp_shape = (28,28)
                cmap = 'gray'

            #if dec: # if p_val is less then significance level: True
            if sample == 1000:
                # store max landmark values
                #np.ones(( len(shifts), random_runs, len(dr_techniques_plot)))
                for dr_idx, dr in enumerate(dr_techniques_plot):
                    #samples_shifts_rands_dr_tech[:,shift_idx,:,dr_idx] = rand_run_p_vals[:,dr_idx,:]
                    max_landmark_values[dr_idx, shift_idx] = landmark_sort_values[dr_idx,0]
                    max_landmark_p_values[dr_idx, shift_idx] = rand_run_p_vals[si,dr_idx,rand_run]
                
                # for dr_ind, dr_technique in enumerate(dr_techniques):
                #     #most_conf_test_indices = test_indices[dr_check,:] # test_indices: sorted landmark indices
                #     current_dr_technique = DimensionalityReduction(dr_technique).name

                #     top_same_samples_path = sample_path + 'top_same/' + current_dr_technique 
                #     if not os.path.exists(top_same_samples_path):
                #         os.makedirs(top_same_samples_path)

                #     rev_top_test_ind = test_indices[dr_ind,:][:difference_samples]
                #     least_conf_samples = Z[rev_top_test_ind]
                #     for j in range(len(rev_top_test_ind)):
                #         samp = least_conf_samples[j, :]
                #         fig = plt.imshow(samp.reshape(samp_shape), cmap=cmap)
                #         plt.axis('off')
                #         fig.axes.get_xaxis().set_visible(False)
                #         fig.axes.get_yaxis().set_visible(False)
                #         plt.savefig("%s/topsame_%s_landmark_alpha%s_%s_%s_%s.pdf" % (top_same_samples_path,datset,sign_level,current_dr_technique,shift, j), 
                #                                                                         bbox_inches='tight', pad_inches=0)
                #         plt.clf()

                #         j = j + 1

                #     top_different_samples_path = sample_path + 'top_diff/' + current_dr_technique 
                #     if not os.path.exists(top_different_samples_path):
                #         os.makedirs(top_different_samples_path)

                #     diff_test_indices = test_indices[dr_ind,:][-difference_samples:]
                #     most_conf_samples = Z[diff_test_indices]
                #     for j in range(len(most_conf_samples)):
                #         if j < difference_samples:
                #             samp = most_conf_samples[j,:]
                #             fig = plt.imshow(samp.reshape(samp_shape), cmap=cmap)
                #             plt.axis('off')
                #             fig.axes.get_xaxis().set_visible(False)
                #             fig.axes.get_yaxis().set_visible(False)
                #             plt.savefig("%s/topdiff_%s_landmark_alpha%s_%s_%s_%s.pdf" % (top_different_samples_path,datset,sign_level,current_dr_technique,shift, j), 
                #                                                                             bbox_inches='tight', pad_inches=0)
                #             plt.clf()

                #             j = j + 1
                #         else:
                #             break
########################################### suppose to be in one dimension above


                    # most_conf_samples = X_te_dcl[most_conf_test_indices]
                    # original_indices = []
                    # j = 0
                    # for i in range(len(most_conf_samples)):
                    #     samp = most_conf_samples[i,:]
                    #     ind = np.where(np.all(X_te_3==samp,axis=1))
                    #     if len(ind[0]) > 0:
                    #         original_indices.append(np.asscalar(ind[0]))
                    #
                    #         if j < difference_samples:
                    #             fig = plt.imshow(samp.reshape(samp_shape), cmap=cmap)
                    #             plt.axis('off')
                    #             fig.axes.get_xaxis().set_visible(False)
                    #             fig.axes.get_yaxis().set_visible(False)
                    #             plt.savefig("%s/%s.pdf" % (top_different_samples_path,j), bbox_inches='tight',
                    #                         pad_inches = 0)
                    #             plt.clf()
                    #
                    #             j = j + 1

# plot top-10 similar & dissimilar sorted landmark values
        # dr_tech = 5
        # dr_index = 5
        # #DimensionalityReduction(dr).name
        # #for dr_idx, dr in enumerate(dr_techniques_plot):
        # plt.plot( test_perc[dr_tech,:][: difference_samples], format[dr_index],  color="blue", label="similar" )
        # plt.plot( test_perc[dr_tech,:][-difference_samples:], format[dr_index], color="black",  label="dissimilar" )
        # plt.show()
        # plt.semilogy()
        # plt.axis("off")
        # plt.legend()
        # plt.savefig("%s/dr_top_values.pdf" % rand_run_path, bbox_inches='tight')
        # plt.clf()
        
        # K.clear_session()
        #rand_run_p_vals[si,:,rand_run] = np.append(ind_od_p_vals.flatten(), p_val)
        #rand_run_p_vals[si,:,rand_run] = ind_od_p_vals.flatten()
        for dr_idx, dr in enumerate(dr_techniques_plot):
            plt.semilogx(np.array(samples), rand_run_p_vals[:,dr_idx,rand_run], format[dr], color=colors[dr], label="%s" % DimensionalityReduction(dr).name)
        plt.axhline(y=sign_level, color='k')
        plt.xlabel('Number of samples')
        plt.ylabel('$p$-value')
        plt.ylim([0, 1])
        plt.savefig("%s/dr_sample_comp_noleg.pdf" % rand_run_path, bbox_inches='tight')
        plt.legend()
        plt.savefig("%s/dr_sample_comp.pdf" % rand_run_path, bbox_inches='tight')
        plt.clf()

        np.savetxt("%s/dr_method_p_vals.csv" % rand_run_path, rand_run_p_vals[:,:,rand_run], delimiter=",")

        #np.random.seed(seed)
        #set_random_seed(seed)
        tf.random.set_seed(seed)

    mean_p_vals = np.mean(rand_run_p_vals, axis=2)
    std_p_vals = np.std(rand_run_p_vals, axis=2)

    mean_val_accs = np.mean(val_accs, axis=0)
    std_val_accs = np.std(val_accs, axis=0)

    mean_te_accs = np.mean(te_accs, axis=0)
    std_te_accs = np.std(te_accs, axis=0)

    if calc_acc and test_type == 'univ':
        mean_dcl_accs = []
        std_dcl_accs = []
        for si, sample in enumerate(samples):
            avg_val = 0
            elem_count = 0
            elem_list = []
            for rand_run in range(random_runs):
                current_val = dcl_accs[si, rand_run]
                if current_val == -1:
                    continue
                elem_list.append(current_val)
                avg_val = avg_val + current_val
                elem_count = elem_count + 1
            std_dcl_accs.append(np.std(np.array(elem_list)))
            if elem_count > 1:
                avg_val = avg_val / elem_count
            else:
                avg_val = -1
            mean_dcl_accs.append(avg_val)

        mean_dcl_accs = np.array(mean_dcl_accs)
        std_dcl_accs = np.array(std_dcl_accs)
        smpl_array = np.array(samples)
        min_one_indices = np.where(mean_dcl_accs == -1)

        print("mean_dcl_accs: ", mean_dcl_accs)
        print("std_dcl_accs: ", std_dcl_accs)
        print("smpl_array: ", smpl_array)

        print("-----------------")

        smpl_array = np.delete(smpl_array, min_one_indices)
        mean_dcl_accs = np.delete(mean_dcl_accs, min_one_indices)
        std_dcl_accs = np.delete(std_dcl_accs, min_one_indices)

        print("mean_dcl_accs: ", mean_dcl_accs)
        print("std_dcl_accs: ", std_dcl_accs)
        print("smpl_array: ", smpl_array)

        accs = np.ones((4, len(samples))) * (-1)
        accs[0] = mean_val_accs
        accs[1] = std_val_accs
        accs[2] = mean_te_accs
        accs[3] = std_te_accs

        dcl_accs = np.ones((3, len(smpl_array))) * (-1)
        dcl_accs[0] = smpl_array
        dcl_accs[1] = mean_dcl_accs
        dcl_accs[2] = std_dcl_accs

        np.savetxt("%s/accs.csv" % shift_path, accs, delimiter=",")
        np.savetxt("%s/dcl_accs.csv" % shift_path, dcl_accs, delimiter=",")

        errorfill(np.array(samples), mean_val_accs, std_val_accs, fmt='-o', color=colors[0], label=r"$p$")
        errorfill(np.array(samples), mean_te_accs, std_te_accs, fmt='-s', color=colors[3], label=r"$q$")
        if len(smpl_array) > 0:
            errorfill(smpl_array, mean_dcl_accs, std_dcl_accs, fmt='--X', color=colors[7], label=r"Classif")
        plt.xlabel('Number of samples')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.savefig("%s/accs.pdf" % shift_path, bbox_inches='tight')
        plt.legend()
        plt.savefig("%s/accs_leg.pdf" % shift_path, bbox_inches='tight')
        plt.clf()
    

    for dr_idx, dr in enumerate(dr_techniques_plot):
        errorfill(np.array(samples), mean_p_vals[:,dr_idx], std_p_vals[:,dr_idx], fmt=format[dr], color=colors[dr], label="%s" % DimensionalityReduction(dr).name)
    plt.axhline(y=sign_level, color='k')
    plt.xlabel('Number of samples')
    plt.ylabel('$p$-value')
    plt.ylim([0, 1])
    plt.savefig("%s/dr_sample_comp_noleg.pdf" % shift_path, bbox_inches='tight')
    plt.legend()
    plt.savefig("%s/fail_%s_landmark_alpha%s_%s_run%s.pdf" % (shift_path,datset,sign_level,shift,random_runs), bbox_inches='tight')
    # logscale
    # plt.ylim([0.001, 1])
    # plt.semilogy()
    # plt.savefig("%s/fail_%s_landmark_alpha%s_%s_run%s_logy.pdf" % (shift_path,datset,sign_level,shift,random_runs), bbox_inches='tight')

    for dr_idx, dr in enumerate(dr_techniques_plot):
        errorfill(np.array(samples), mean_p_vals[:,dr_idx], std_p_vals[:,dr_idx], fmt=format[dr], color=colors[dr])
        plt.xlabel('Number of samples')
        plt.ylabel('$p$-value')
        plt.axhline(y=sign_level, color='k', label='sign_level')
        plt.ylim([0, 1])
        plt.savefig("%s/%s_conf.pdf" % (shift_path, DimensionalityReduction(dr).name), bbox_inches='tight')
        plt.clf()

    np.savetxt("%s/mean_p_vals.csv" % shift_path, mean_p_vals, delimiter=",")
    np.savetxt("%s/std_p_vals.csv" % shift_path, std_p_vals, delimiter=",")

    for dr_idx, dr in enumerate(dr_techniques_plot):
        samples_shifts_rands_dr_tech[:,shift_idx,:,dr_idx] = rand_run_p_vals[:,dr_idx,:]

    np.save("%s/samples_shifts_rands_dr_tech.npy" % (path), samples_shifts_rands_dr_tech)

for dr_idx, dr in enumerate(dr_techniques_plot):
    dr_method_results = samples_shifts_rands_dr_tech[:,:,:,dr_idx]

    mean_p_vals = np.mean(dr_method_results, axis=2)
    std_p_vals = np.std(dr_method_results, axis=2)

    for idx, shift in enumerate(shifts):
        errorfill(np.array(samples), mean_p_vals[:, idx], std_p_vals[:, idx], fmt=linestyles[idx]+markers[dr], color=colorscale(colors[dr],brightness[idx]), label="%s" % shift.replace('_', '\\_'))
    plt.xlabel('Number of samples')
    plt.ylabel('$p$-value')
    plt.axhline(y=sign_level, color='k')
    plt.ylim([0, 1])
    plt.savefig("%s/%s_conf_noleg.pdf" % (path, DimensionalityReduction(dr).name), bbox_inches='tight')
    plt.legend()
    plt.savefig("%s/%s_conf.pdf" % (path, DimensionalityReduction(dr).name), bbox_inches='tight')
    plt.clf()

np.save("%s/samples_shifts_rands_dr_tech.npy" % (path), samples_shifts_rands_dr_tech)




shifts = ['small_gn_shift_0.1',
              'small_gn_shift_0.5',
              'small_gn_shift_1.0']

dr_techniques = ["small_gn_shift_0.1", "small_gn_shift_0.5", "small_gn_shift_1.0"]
shift_level = ["NoRed", "PCA", "SRP","UAE", "TAE", "BBSDs"]
harvest = max_landmark_values

fig, ax = plt.subplots()
im = ax.imshow(max_landmark_values)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(dr_techniques)), labels=dr_techniques)
ax.set_yticks(np.arange(len(shift_level)), labels=shift_level)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(shift_level)):
    for j in range(len(dr_techniques)):
        text = ax.text(j, i, max_landmark_values[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Gaussian noise added")
fig.tight_layout()
plt.xlabel('values')
plt.ylabel('some values')
#plt.savefig("%s/z_%s_table.pdf" % (shift_path, shift_type), bbox_inches='tight')
#plt.clf()

#np.savetxt("%s/dr_method_p_vals.csv" % rand_run_path, rand_run_p_vals[:,:,rand_run], delimiter=",")
#create path for csv tables
path2 = path1
path2 += 'pval_csv_tables' + '/'
if not os.path.exists(path2):
    os.makedirs(path2)
    
np.savetxt("%s/%s_%s_%s_max_landmark_pval.csv" % (path2,datset,sign_level,shift_type), max_landmark_p_values, delimiter=",")


#create path for csv tables  
path3 = path1
path3 += 'lval_csv_tables' + '/'
if not os.path.exists(path3):
    os.makedirs(path3)
    
np.savetxt("%s/%s_%s_%s_max_landmark.csv" % (path3,datset,sign_level,shift_type), max_landmark_values, delimiter=",")
