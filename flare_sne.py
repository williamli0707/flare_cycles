import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
from tqdm import tqdm
from sklearn.manifold import TSNE

NAMES = ("t_start", "t_stop", "t_peak", "amplitude", "FWHM", "duration", "t_peak_aflare1", "t_FWHM_aflare1", "amplitude_aflare1",
"flare_chisq", "KS_d_model", "KS_p_model", "KS_d_cont", "KS_p_cont", "Equiv_Dur", "ED68i", "ED90i")

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_columns', None)

def get_size(file):
    temp = pd.read_table(file, names=['kic'])
    get_size_col = temp['kic'].values
    return get_size_col.size

def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range

def main(BIN_LENGTH, NUM_DAYS_INPUT):
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

    inputs = []

    for line in tqdm(targets, desc="Loading data", total=target_count):
        data_energy = []
        data_date = []
        data_duration = []
        data_amplitude = []
        data_fwhm = []
        data_flare_chisq = []

        # if ind: break
        KIC = line.rstrip('\n')
        files = sorted(glob('KICs/' + KIC + "/*.flare"))
        num_files = len(files)

        for x in range(num_files):
            df = pd.read_table(files[x], comment="#", delimiter=",", names=NAMES)
            # grabbing the energy (equivalent duration) column from each file, sorting, then including only the positive values so it can be logged
            energy = np.array(df['Equiv_Dur'])
            positive = np.where(energy > 0)

            # print("processing file", files[x], "with", len(energy), "energy values and", len(positive[0]), "positive energy values")

            data_energy += [energy[i] for i in positive[0]]
            data_date += [np.array(df['t_start'])[i] for i in positive[0]]
            data_duration += [np.array(df['duration'])[i] for i in positive[0]]
            data_amplitude += [np.array(df['amplitude'])[i] for i in positive[0]]
            data_fwhm += [np.array(df['FWHM'])[i] for i in positive[0]]
            data_flare_chisq += [np.array(df['flare_chisq'])[i] for i in positive[0]]

        # print("should be equal:")
        data = zip(data_date, data_energy, data_duration, data_amplitude, data_fwhm, data_flare_chisq)
        data = np.array(sorted(data))
        input_case = []

        for index in range(0, len(data) - 1):
            i = data[index]
            input_case.append(i)
            while i[0] - input_case[0][0] > NUM_DAYS_INPUT: input_case.pop(0)
            if len(input_case) >= MIN_NUM_INPUTS and data[index + 1][0] - i[0] <= 10:
                use = True
                tmpind = index + 1
                if tmpind >= len(data): break
                while tmpind < len(data) and data[tmpind][0] - i[0] <= 10: tmpind += 1
                # binning:
                input_case_insert = []
                start_date = input_case[0][0]
                for bin1 in range(0, int(NUM_DAYS_INPUT / BIN_LENGTH)): #bin1 because bin is a keyword
                    bin_start = start_date + bin1 * BIN_LENGTH
                    bin_end = bin_start + BIN_LENGTH
                    bin_data = [x for x in input_case if bin_start <= x[0] < bin_end]
                    if len(bin_data) == 0:
                        use = False
                        # print("aborting after", bin1, "bins because there was no data in bin", bin_start, "to", bin_end)
                        ctr += 1
                        break
                        # input_case_insert.append(0)
                    else:
                        input_case_insert.append(np.average(np.array(bin_data)[:, 1], weights=np.array(bin_data)[:, 2]))
                if not use: continue
                inputs.append(input_case_insert)
                break

        ind += 1

    # print(inputs)
    inputs = np.array(inputs)
    print('done loading data')
    print(ctr, "times we had to skip a bin because there was insufficient data in it")
    print("checking for nan:", np.argwhere(np.isnan(inputs)))
    print("checking for inf:", np.argwhere(np.isinf(inputs)))

    # inputs = np.random.randint(low=1000, high=2000, size=(167441, int(NUM_DAYS_INPUT/BIN_LENGTH)))

    model = tf.keras.Sequential([
        # tf.keras.layers.Input(shape=[None], ragged=True),
        tf.keras.layers.Dense(128, activation='relu', input_shape=(int(NUM_DAYS_INPUT / BIN_LENGTH),)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1, activation='relu')
    ])

    print('model instantiated')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )

    print('model compiled')
    print('length of inputs', len(inputs))
    print('shape of inputs', inputs.shape)

    history = model.fit(
        inputs,
        epochs=40,
        verbose=1,
    )

    model2 = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    test_ds = np.concatenate(
        list(inputs.take(5).map(lambda x, y: x)))  # get five batches of images and convert to numpy array
    features = model2(test_ds)
    labels = np.argmax(model(test_ds), axis=-1)
    tsne = TSNE(n_components=2).fit_transform(features)
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    current_tx = np.take(tx)
    current_ty = np.take(ty)
    ax.scatter(current_tx, current_ty)

    ax.legend(loc='best')
    plt.show()

    return inputs, model, history