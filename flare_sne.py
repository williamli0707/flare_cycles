import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import itertools
from tqdm import tqdm
import time

NAMES = ("t_start", "t_stop", "t_peak", "amplitude", "FWHM", "duration", "t_peak_aflare1", "t_FWHM_aflare1", "amplitude_aflare1",
"flare_chisq", "KS_d_model", "KS_p_model", "KS_d_cont", "KS_p_cont", "Equiv_Dur", "ED68i", "ED90i")

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_columns', None)

def get_size(file):
    temp = pd.read_table(file, names=['kic'])
    get_size_col = temp['kic'].values
    return get_size_col.size

file = 'targets/targets_full.txt'
target_count = get_size(file)
targets = open(file, "r")

data = []

for line in tqdm(targets, desc="Loading data", total=target_count):
    KIC = line.rstrip('\n')
    files = sorted(glob('KICs/' + KIC + "/*.flare"))
    num_files = len(files)
    data.append([])

    for x in range(num_files):
        df = pd.read_table(files[x], comment="#", delimiter=",", names=NAMES)