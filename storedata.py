import numpy as np
import sys
import pandas as pd
from glob import glob
from tqdm import tqdm
import os

NAMES = ("t_start", "t_stop", "t_peak", "amplitude", "FWHM", "duration", "t_peak_aflare1", "t_FWHM_aflare1", "amplitude_aflare1",
"flare_chisq", "KS_d_model", "KS_p_model", "KS_d_cont", "KS_p_cont", "Equiv_Dur", "ED68i", "ED90i")

def get_size(file):
    temp = pd.read_table(file, names=['kic'])
    get_size_col = temp['kic'].values
    return get_size_col.size

def store_time(BIN_LENGTH, NUM_DAYS_INPUT):
    MIN_NUM_INPUTS = 50

    assert NUM_DAYS_INPUT % BIN_LENGTH == 0

    np.set_printoptions(threshold=sys.maxsize)
    pd.set_option('display.max_columns', None)
    # file = 'targets/target_single.txt'
    file = 'targets/targets_full.txt'
    target_count = get_size(file)
    targets = open(file, "r")

    ind = 0
    ctr = 0
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
                start_date = input_case[0][0]
                for bin1 in range(0, int(NUM_DAYS_INPUT / BIN_LENGTH)): #bin1 because bin is a keyword
                    bin_start = start_date + bin1 * BIN_LENGTH
                    bin_end = bin_start + BIN_LENGTH
                    bin_data = [x for x in input_case if bin_start <= x[0] < bin_end]
                    if len(bin_data) == 0:
                        # use = False
                        # # print("aborting after", bin1, "bins because there was no data in bin", bin_start, "to", bin_end)
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
    # print(ctr, "times we had to skip a bin because there was insufficient data in it")
    print("checking for nan:", np.argwhere(np.isnan(inputs)))
    print("checking for inf:", np.argwhere(np.isinf(inputs)))

    if not os.path.exists('./data'):
        os.mkdir('./data')

    np.savez('./data/time_' + str(BIN_LENGTH) + '_' + str(NUM_DAYS_INPUT) + '.npz', inputs=inputs, outputs=outputs)

def store_time_nd(BIN_LENGTH, NUM_DAYS_INPUT):
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
    ctr = 0
    data = []

    inputs = []
    outputs = []

    for line in tqdm(targets, desc="Loading data", total=target_count):
        power_data_ed = []
        power_data_dates = []

        # if ind: break
        KIC = line.rstrip('\n')
        files = sorted(glob('KICs/' + KIC + "/*.flare"))
        num_files = len(files)
        data.append([])

        for x in range(num_files):
            df = pd.read_table(files[x], comment="#", delimiter=",", names=NAMES)
            # grabbing the energy (equivalent duration) column from each file, sorting, then including only the positive values so it can be logged
            ed = np.array(df['Equiv_Dur'])
            start = np.array(df['t_start'])

            # print("processing file", files[x], "with", len(energy), "energy values and", len(positive[0]), "positive energy values")

            power_data_ed += ed.tolist()
            power_data_dates += start.tolist()

        # print("should be equal:")
        power_data = zip(power_data_dates, power_data_ed)
        power_data = np.array(sorted(power_data))
        input_case = []

        for index, i in enumerate(power_data):
            if index == len(power_data) - 1: break
            input_case.append(i)
            while i[0] - input_case[0][0] > NUM_DAYS_INPUT: input_case.pop(0)
            if len(input_case) >= MIN_NUM_INPUTS and power_data[index + 1][0] - i[0] <= 10:
                use = True
                tmpind = index + 1
                if tmpind >= len(power_data): break
                while tmpind < len(power_data) and power_data[tmpind][0] - i[0] <= 10: tmpind += 1
                tmp = power_data[index + 1:tmpind, :]  # for output
                # binning:
                input_case_insert = []
                start_date = input_case[0][0]
                for bin1 in range(0, int(NUM_DAYS_INPUT / BIN_LENGTH)):  # bin1 because bin is a keyword
                    bin_start = start_date + bin1 * BIN_LENGTH
                    bin_end = bin_start + BIN_LENGTH
                    bin_data = [x for x in input_case if bin_start <= x[0] < bin_end]
                    if len(bin_data) == 0:
                        # use = False
                        # # print("aborting after", bin1, "bins because there was no data in bin", bin_start, "to", bin_end)
                        # ctr += 1
                        # break
                        input_case_insert.append(0)
                    else:
                        input_case_insert.append(np.average(np.array(bin_data)[:, 1]))
                if not use: continue
                inputs.append(input_case_insert)
                outputs.append(np.average(tmp[:, 1]))
                # the output should represent the average of power output over next 10 days -> weighted average of next 10 days of stuff

        ind += 1

    # print(inputs)
    # print(outputs)
    print('done loading data')
    # print(ctr, "times we had to skip a bin because there was insufficient data in it")
    print("checking for nan:", np.argwhere(np.isnan(inputs)))
    print("checking for inf:", np.argwhere(np.isinf(inputs)))

    if not os.path.exists('./data'):
        os.mkdir('./data')

    print("saving to " + os.getcwd() + '/data/time_nd_' + str(BIN_LENGTH) + '_' + str(NUM_DAYS_INPUT) + '.npz')
    np.savez(os.getcwd() + '/data/time_nd_' + str(BIN_LENGTH) + '_' + str(NUM_DAYS_INPUT) + '.npz', inputs=inputs, outputs=outputs)

def store_smooth(BIN_LENGTH, NUM_DAYS_INPUT):
    MIN_NUM_INPUTS = 50

    assert NUM_DAYS_INPUT % BIN_LENGTH == 0

    np.set_printoptions(threshold=sys.maxsize)
    pd.set_option('display.max_columns', None)
    # file = 'targets/target_single.txt'
    file = 'targets/targets_full.txt'
    target_count = get_size(file)
    targets = open(file, "r")

    ind = 0
    ctr = 0
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

        for index in range(0, len(power_data) - 1):
            i = power_data[index]
            input_case.append(i)
            while i[0] - input_case[0][0] > NUM_DAYS_INPUT: input_case.pop(0)
            if len(input_case) >= MIN_NUM_INPUTS and power_data[index + 1][0] - i[0] <= 10:
                use = True
                tmpind = index + 1
                if tmpind >= len(power_data): break
                while tmpind < len(power_data) and power_data[tmpind][0] - i[0] <= 10: tmpind += 1
                tmp = power_data[index + 1:tmpind, :]  # for output
                # binning:
                input_case_insert = []
                start_date = input_case[0][0]
                for bin1 in range(0, int(NUM_DAYS_INPUT / BIN_LENGTH)):  # bin1 because bin is a keyword
                    bin_start = start_date + bin1 * BIN_LENGTH
                    bin_end = bin_start + BIN_LENGTH
                    bin_data = [x for x in input_case if bin_start <= x[0] < bin_end]
                    if len(bin_data) == 0:
                        # use = False
                        # # print("aborting after", bin1, "bins because there was no data in bin", bin_start, "to", bin_end)
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
    # print(ctr, "times we had to skip a bin because there was insufficient data in it")
    print("checking for nan:", np.argwhere(np.isnan(inputs)))
    print("checking for inf:", np.argwhere(np.isinf(inputs)))

    if not os.path.exists('./data'):
        os.mkdir('./data')

    np.savez('./data/time_' + str(BIN_LENGTH) + '_' + str(NUM_DAYS_INPUT) + '.npz', inputs=inputs, outputs=outputs)