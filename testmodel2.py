import tensorflow as tf
import numpy as np

inputs = np.random.randint(low=1000, high=2000, size=(5000, 300))
outputs = np.random.randint(low=2000, high=2500, size=(5000,))

model = tf.keras.Sequential([
  tf.keras.layers.Dense(128, activation='relu', input_shape=(300,)),
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
  epochs=1000,
  verbose=1,
  validation_split=0.1
)


