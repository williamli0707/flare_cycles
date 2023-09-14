import matplotlib as plt
import numpy as np
import pandas as pd
from flare_model_time_binning import main

bin_sizes = [2, 10, 20, 50]
for i in bin_sizes:
    inputs, outputs, model, history = main(bin_sizes, 100)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')