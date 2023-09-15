import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flare_model_time_binning import main

bin_sizes = [2, 10, 20, 50]
for i in bin_sizes:
    inputs, outputs, model, history = main(i, 100)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

plt.title('model loss per epoch by bin size')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend([('train_' + str(i)) for i in bin_sizes] + [('val_' + str(i)) for i in bin_sizes], loc='upper left')
plt.show()
#