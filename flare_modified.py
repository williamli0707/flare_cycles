import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import sys
import tensorflow as tf

NAMES=("t_start", "t_stop", "t_peak", "amplitude", "FWHM", "duration", "t_peak_aflare1", "t_FWHM_aflare1", "amplitude_aflare1", "flare_chisq", "KS_d_model", "KS_p_model", "KS_d_cont", "KS_p_cont", "Equiv_Dur", "ED68i", "ED90i")

def get_size(file):
    '''
    Returning the number of KICs in a targettext file.

    Parameters:
    -------------------------------------------------------------------------
    file (string): The path to the file containing the list of KICs.


    Returns:
    -------------------------------------------------------------------------
    get_size_col.size (integer): The number of KICs in the file.
    '''

    temp = pd.read_table(file, names=['kic'])
    get_size_col = temp['kic'].values
    return get_size_col.size


def main():
    np.set_printoptions(threshold=sys.maxsize)
    pd.set_option('display.max_columns', None)
    # file = 'targets/target_single.txt'
    file = 'targets/targets_full.txt'
    target_count = get_size(file)
    targets = open(file, "r")

    tot = 0
    totrows = 0
    x_axis=[]
    y_axis=[]
    ind = 0
    target = 0
    data = []
    for line in targets:
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
            data[ind] = [
                np.array(df['amplitude']),
                np.array(df['FWHM']),
                np.array(df['duration']),
                np.array(df['t_peak_aflare1']),
                np.array(df['t_FWHM_aflare1']),
                np.array(df['t_amplitude_aflare1']),
                np.array(df['flare_chisq']),
                np.array(df['KS_d_model']),
                np.array(df['KS_p_model']),
                np.array(df['KS_d_cont']),
                np.array(df['KS_p_cont']),
                np.array(df['Equiv_Dur'])
            ]
            # print(energy)
            if ind == target:
                print(df)
                print()
                print(energy)
                print()
                print(files[x])

                print("\n\n\n\n\ntest")
                print(positive)

                # for i in range(0, len(df)):
                #     x_axis.append(start[i])
                #     y_axis.append(energy[i] / dur[i])
                #     # y_axis.append(energy[i])
                for i in positive[0]:
                    x_axis.append(start[i])
                    y_axis.append(energy[i] / dur[i])

        if ind == target:
            # y_axis = np.log(y_axis)
            plt.scatter(x_axis, y_axis, c='blue')
            # plt.ylim(-100, 100)
            # plt.title("Energy vs start times for KIC " + line)
            # plt.ylim(-500, 500)
            plt.xlim(400, 800)
            plt.title("Power (+) vs start times for KIC " + line)
            plt.show()

            y_axis = np.log(y_axis)
            plt.title("Power (+) vs start times for KIC " + line + ", logarithmic y scale")
            plt.scatter(x_axis, y_axis, c='blue')
            plt.xlim(400, 800)
            plt.show()


        ind += 1

    print(tot)
    print(totrows)

    print("\n\n\n\n\n")
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255, input_shape=(180, 180, 3)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5)
    ])

main()