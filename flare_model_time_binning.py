import numpy as np
import pandas as pd
from glob import glob
import sys
import tensorflow as tf
from tqdm import tqdm
import time

NAMES = ("t_start", "t_stop", "t_peak", "amplitude", "FWHM", "duration", "t_peak_aflare1", "t_FWHM_aflare1", "amplitude_aflare1",
"flare_chisq", "KS_d_model", "KS_p_model", "KS_d_cont", "KS_p_cont", "Equiv_Dur", "ED68i", "ED90i")



def get_size(file):
    temp = pd.read_table(file, names=['kic'])
    get_size_col = temp['kic'].values
    return get_size_col.size

def main(BIN_LENGTH, NUM_DAYS_INPUT):
    # NUM_DAYS_INPUT = 100
    MIN_NUM_INPUTS = 50
    # BIN_LENGTH=10 # days
    # BIN_LENGTH = input("Enter bin length in days: ")
    MIN_FUTURE_OUTPUT = 50

    assert NUM_DAYS_INPUT % BIN_LENGTH == 0

    np.set_printoptions(threshold=sys.maxsize)
    pd.set_option('display.max_columns', None)
    # file = 'targets/target_single.txt'
    file = 'targets/targets_full.txt'
    target_count = get_size(file)
    targets = open(file, "r")

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
        ctr = 0

        for index in range(0, len(power_data) - 1):
            i = power_data[index]
            input_case.append(i)
            while i[0] - input_case[0][0] > NUM_DAYS_INPUT: input_case.pop(0)
            if len(input_case) >= MIN_NUM_INPUTS and power_data[index + 1][0] - i[0] <= 10:
                use = True
                tmpind = index + 1
                if tmpind >= len(power_data): break
                while tmpind < len(power_data) and power_data[tmpind][0] - i[0] <= 10: tmpind += 1
                tmp = power_data[index + 1:tmpind, :] # for output
                # binning:
                input_case_insert = []
                start_date = power_data[index + 1][0]
                for bin1 in range(0, int(NUM_DAYS_INPUT / BIN_LENGTH)): #bin1 because bin is a keyword
                    bin_start = start_date + bin1 * BIN_LENGTH
                    bin_end = bin_start + BIN_LENGTH
                    bin_data = [x for x in input_case if bin_start <= x[0] < bin_end]
                    if len(bin_data) == 0:
                        # use = False
                        # ctr += 1
                        # break
                        input_case_insert.append(0)
                    else:
                        input_case_insert.append(np.average(np.array(bin_data)[:, 1], weights=np.array(bin_data)[:, 2]))
                if not use: continue
                inputs.append(input_case_insert)
                outputs.append(np.average(tmp[:, 1], weights=tmp[:, 2]))
                # the output should represent the average of power output over next 10 days -> weighted average of next 10 days of stuff

        ind += 1

    # print(inputs)
    # print(outputs)
    print('done loading data')
    print(ctr, "times we had to skip a bin because there was no data in it")

    # inputs = np.random.randint(low=1000, high=2000, size=(167441, NUM_DAYS_INPUT))
    # outputs = np.random.randint(low=2000, high=2500, size=(167441,))

    # inputs = np.array(inputs)[0:5000, :]
    # outputs = np.array(outputs)[0:5000]

    model = tf.keras.Sequential([
        # tf.keras.layers.Input(shape=[None], ragged=True),
        tf.keras.layers.Dense(128, activation='relu', input_shape=(int(NUM_DAYS_INPUT / BIN_LENGTH),)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1, activation='relu')
    ])

    print('model instantiated')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error')

    print('model compiled')
    print('length of inputs', len(inputs))

    history = model.fit(
        # tf.ragged.constant(inputs),
        np.array(inputs),
        np.array(outputs),
        epochs=40,
        # Suppress logging.
        verbose=1,
        # Calculate validation results on 20% of the training data.
        validation_split=0.1,
        batch_size=50
    )

    tot = 0
    for i in range(0, len(inputs)):
        ctot = 0
        for j in inputs[i]: ctot += j
        ctot /= len(inputs[i])
        tot += (outputs[i] - ctot) ** 2

    print("MSE for average of inputs:", tot / len(inputs))

    return inputs, outputs, model, history