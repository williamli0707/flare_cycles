import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import itertools
from tqdm import tqdm
import time

NAMES = (
"t_start", "t_stop", "t_peak", "amplitude", "FWHM", "duration", "t_peak_aflare1", "t_FWHM_aflare1", "amplitude_aflare1",
"flare_chisq", "KS_d_model", "KS_p_model", "KS_d_cont", "KS_p_cont", "Equiv_Dur", "ED68i", "ED90i")

NUM_DAYS_INPUT = 300
MIN_NUM_INPUTS = 50

NUM_IN_INPUTS = 400
MIN_FUTURE_OUTPUT = 50


def get_size(file):
    temp = pd.read_table(file, names=['kic'])
    get_size_col = temp['kic'].values
    return get_size_col.size



np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_columns', None)
# file = 'targets/target_single.txt'
file = 'targets/targets_full.txt'
target_count = get_size(file)
targets = open(file, "r")

tot = 0
totrows = 0
ind = 0
data = []

inputs = []
outputs = []

for line in tqdm(targets, desc="Loading data", total=target_count):
    power_data_powers = []
    power_data_dates = []
    power_data_durations = []

    # if ind: break
    KIC = line.rstrip('\n')
    files = sorted(glob('KICs/' + KIC + "/*.flare"))
    num_files = len(files)
    data.append([])

    for x in range(num_files):
        df = pd.read_table(files[x], comment="#", delimiter=",", names=NAMES)
        tot += df.size
        totrows += len(df.index)
        # grabbing the energy (equivalent duration) column from each file, sorting, then including only the positive values so it can be logged
        energy = np.array(df['Equiv_Dur'])
        positive = np.where(energy > 0)
        start = np.array(df['t_start'])
        dur = np.array(df['duration'])

        # print("processing file", files[x], "with", len(energy), "energy values and", len(positive[0]), "positive energy values")

        power_data_durations += [dur[i] for i in positive[0]]
        power_data_powers += [(energy[i] / dur[i]) for i in positive[0]]
        power_data_dates += [start[i] for i in positive[0]]

    # print("should be equal:")
    power_data = zip(power_data_dates, power_data_powers, power_data_durations)
    power_data = np.array(sorted(power_data))
    input_case = []

    for index in range(0, len(power_data) - MIN_FUTURE_OUTPUT):
        i = power_data[index]
        input_case.append(i[0])
        if len(input_case) > NUM_DAYS_INPUT:
            input_case.pop(0)
            tmp = power_data[index + 1:index + 1 + MIN_FUTURE_OUTPUT, :]
            inputs.append(input_case)
            outputs.append(np.average(tmp[:, 1], weights=tmp[:, 2]))

    '''
    # old code for using window of time as input
    for index in range(0, len(power_data) - 1):
        i = power_data[index]
        input_case.append(i[0])
        if i[0] - input_case[0] > NUM_DAYS_INPUT: input_case.pop(0)
        if len(input_case) >= MIN_NUM_INPUTS and power_data[index + 1][0] - i[0] <= 10:
            tmpind = index + 1
            if tmpind >= len(power_data): break
            while tmpind < len(power_data) and power_data[tmpind][0] - i[0] <= 10: tmpind += 1
            tmp = power_data[index + 1:tmpind, :]
            # print("shape of tmp", np.array(tmp).shape, index + 1, tmpind)
            # print(tuple(tmp))
            # print(np.array([i[1] for i in tmp]).shape, np.array([i[2] for i in tmp]).shape)
            inputs.append(input_case)
            outputs.append(np.average(tmp[:, 1], weights=tmp[:, 2]))
            # the output should represent the average of power output over next 10 days -> weighted average of next 10 days of stuff
    '''
    ind += 1

# print(inputs)
# print(outputs)
print('done loading data')

model = tf.keras.Sequential([
    # tf.keras.layers.Input(shape=[None], ragged=True),
    tf.keras.layers.Dense(128, activation='relu', input_shape=(NUM_DAYS_INPUT,)),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1, activation='relu')
])

print('model instantiated')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

print('model compiled')
print('length of inputs', len(inputs))

history = model.fit(
    # tf.ragged.constant(inputs),
    np.array(inputs),
    np.array(outputs),
    epochs=1,
    # Suppress logging.
    verbose=1,
    # Calculate validation results on 20% of the training data.
    validation_split=0.1)