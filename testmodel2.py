import numpy as np
import tensorflow as tf

NUM_DAYS_INPUT = 300

inputs = np.random.randint(low=1000, high=2000, size=(200000, NUM_DAYS_INPUT))
outputs = np.random.randint(low=2000, high=2500, size=(200000,))

# inputs = np.array(inputs)[0:5000, :]
# outputs = np.array(outputs)[0:5000]

model = tf.keras.Sequential([
    # tf.keras.layers.Input(shape=[None], ragged=True),
    tf.keras.layers.Dense(128, activation='relu', input_shape=(NUM_DAYS_INPUT,)),
    tf.keras.layers.Dense(1024, activation='relu'),
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
    epochs=20,
    # Suppress logging.
    verbose=1,
    # Calculate validation results on 20% of the training data.
    validation_split=0.1,
    batch_size=50
)