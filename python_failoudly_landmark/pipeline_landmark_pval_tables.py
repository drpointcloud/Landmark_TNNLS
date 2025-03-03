# get from https://github.com/steverab/failing-loudly
# may/might have been modified
# -------------------------------------------------
# IMPORTS
# sample size for multiv is changed[line 140s]
# the mpl.rParams is changed[line 40s]
# -------------------------------------------------

from urllib.request import pathname2url
import numpy as np

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


make_keras_picklable()
# -------------------------------------------------
# CONFIG
# -------------------------------------------------
# Number of random runs to average results over.
random_runs = 250

# change line 121, 122, and [148 149] accordingly
# hypothesis_test = "MMD"
# sys.argv = [ 'pipeline_landmark.py', 'cifar10', 'small_image_shift', 'multiv', 'LMSW']
datset = 'mnist'
#shift_type = sys.argv[2]

shift =  'ko_shift_0.5' # 'ko_shift_0.1',  'ko_shift_0.5',     'ko_shift_1.0'


test_type = 'multiv'


# Define results path and create directory.
path = './results/'
path += test_type + '/'

#create path for csv tables
path1 = path
if not os.path.exists(path1):
    os.makedirs(path1)


dr_techniques = [DimensionalityReduction.NoRed.value]#, DimensionalityReduction.BBSDs.value]
dr_technique_names = ['Orig']
md_test_names = ['MMD', 'MMD_bug','MLW']
#md_test_names = ['MMD', 'MMD_bug','LMSW','MLW']
#md_tests = [MultidimensionalTest.MMD.value,MultidimensionalTest.MMD_bug.value, MultidimensionalTest.LMSW.value,  MultidimensionalTest.MLW.value]
md_tests = [MultidimensionalTest.MMD.value,MultidimensionalTest.MMD_bug.value, MultidimensionalTest.MLW.value]

sign_level = 0.1
samples = [200, 400, 700, 1000]

red_dim = -1
red_models = [None] * len(DimensionalityReduction)

# create folders for each shift
shift_path = path1 + shift + '/'
if not os.path.exists(shift_path):
    os.makedirs(shift_path)

# Stores p-values for a single shift.
rand_run_p_vals = np.ones((len(samples), len(dr_techniques),len(md_tests), random_runs)) * (-1)

# Average over a few random runs to quantify robustness.
for rand_run in range(0, random_runs):

    np.random.seed(rand_run)

    # Load data.
    (X_tr_orig, y_tr_orig), (X_val_orig, y_val_orig), (X_te_orig, y_te_orig), orig_dims, nb_classes = \
        import_dataset(datset, shuffle=True) # Mixes MNIST training and testing

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

    X_te_2 , y_te_2 = random_shuffle(X_te_1, y_te_1) # shuffles
    # Check detection performance for different numbers of samples from test.
    for si, sample in enumerate(samples):
        print("%s: %s RandomRun %s/%s SampleSize %s"  % (datset,shift,rand_run+1,random_runs,sample))


        X_te_3 = X_te_2[:sample,:]
        y_te_3 = y_te_2[:sample]

        X_val_3 = X_val_orig[:1000,:]
        y_val_3 = y_val_orig[:1000]

        X_tr_3 = np.copy(X_tr_orig[:1000,:])
        y_tr_3 = np.copy(y_tr_orig[:1000])

        # Detect shift.
        shift_detector = ShiftDetector(dr_techniques, [TestDimensionality.Multi.value],[], md_tests, sign_level, red_models,
                                       sample, datset)

        (od_decs, ind_od_decs, ind_od_p_vals), \
        (md_decs, ind_md_decs, ind_md_p_vals), \
        red_dim, red_models, _, _  = shift_detector.detect_data_shift(X_tr_3, y_tr_3, X_val_3, y_val_3,
                                                                                X_te_3, y_te_3, orig_dims,
                                                                                nb_classes)



        rand_run_p_vals[si,:,:,rand_run] = ind_md_p_vals
        with open(shift_path+f"p_vals_{datset}_{sample}.csv", "ab") as f:
            np.savetxt(f, rand_run_p_vals[si,:,:,rand_run].flatten(), delimiter=",")


for si in range(len(samples)):
    for i, dr_technique in enumerate(dr_technique_names):
        for j, md_test in enumerate(md_test_names):
            sample = rand_run_p_vals[si,i,j, :].flatten()
            np.save(f'{datset}_{dr_technique}_{md_test}.npy', sample)
