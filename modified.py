import keras
import numpy as np
import pandas as pd
from glob import glob
import sys
import tensorflow as tf
from tqdm import tqdm
import time
import os
from storedata import store_time

NAMES = ("t_start", "t_stop", "t_peak", "amplitude", "FWHM", "duration", "t_peak_aflare1", "t_FWHM_aflare1", "amplitude_aflare1",
"flare_chisq", "KS_d_model", "KS_p_model", "KS_d_cont", "KS_p_cont", "Equiv_Dur", "ED68i", "ED90i")



def get_size(file):
    temp = pd.read_table(file, names=['kic'])
    get_size_col = temp['kic'].values
    return get_size_col.size

def main(BIN_LENGTH, NUM_DAYS_INPUT):
    if not os.path.exists('./data/time_' + str(BIN_LENGTH) + '_' + str(NUM_DAYS_INPUT) + '.npz'):
        store_time(BIN_LENGTH, NUM_DAYS_INPUT)

    data = np.load('./data/time_' + str(BIN_LENGTH) + '_' + str(NUM_DAYS_INPUT) + '.npz', allow_pickle=True)
    inputs = data['inputs']
    outputs = data['outputs']

    # inputs = np.random.randint(low=1000, high=2000, size=(167441, int(NUM_DAYS_INPUT/BIN_LENGTH)))
    # outputs = np.random.randint(low=2000, high=2500, size=(167441,))

    # inputs = np.array(inputs)[0:5000, :]
    # outputs = np.array(outputs)[0:5000]

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(inputs))

    model = tf.keras.Sequential([
        # tf.keras.layers.Input(shape=[None], ragged=True),
        normalizer,
        tf.keras.layers.Dense(128, activation='linear', input_shape=(int(NUM_DAYS_INPUT / BIN_LENGTH),)),
        tf.keras.layers.Dense(1024, activation='linear'),
        tf.keras.layers.Dense(2048, activation='linear'),
        tf.keras.layers.Dense(1024, activation='linear'),
        tf.keras.layers.Dense(1, activation='linear')
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
        epochs=2,
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
