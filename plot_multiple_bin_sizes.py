import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flare_model_time_binning import main

bin_sizes = [2, 10, 20, 50]
results = []
for i in bin_sizes:
    inputs, outputs, model, history = main(i, 100)
    results.append((inputs, outputs, model, history))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    cont = input("continue? (y/n)")
    if cont == 'n': break

plt.title('model loss per epoch by bin size')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.yscale("log")
a = [('train_' + str(i)) for i in bin_sizes]
b = [('val_' + str(i)) for i in bin_sizes]
plt.legend(np.vstack((a,b)).reshape((-1,),order='F'), loc='upper left')
plt.show()
#