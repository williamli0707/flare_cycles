import random
import tensorflow as tf
import numpy as np

inputs = []
outputs = []

for i in range(0, 500):
  a = int(random.random() * 10) + 1
  b = int(random.random() * 10) + 1
  c = int(random.random() * 10) + 1
  d = int(random.random() * 10) + 1
  e = int(random.random() * 10) + 1
  inputs.append([a, b, c, d, e])
  outputs.append(a * b * c * d * e)

model = tf.keras.Sequential([
  tf.keras.layers.Dense(128, activation='relu', input_shape=(5,)),
  tf.keras.layers.Dense(1024,activation='relu'),
  tf.keras.layers.Dense(1024,activation='relu'),
  tf.keras.layers.Dense(1,activation='relu')
])

model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
  loss='mean_squared_error'
)

model.fit(
  np.array(inputs),
  np.array(outputs),
  epochs=20,
  verbose=1,
  validation_split=0.1
)


