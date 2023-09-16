import tensorflow as tf
import pathlib

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  seed=123,
  image_size=(180, 180),
  batch_size=batch_size)


model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255, input_shape=(180, 180, 3)),
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

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
epochs=30
history = model.fit(
  train_ds,
  epochs=epochs
)

from sklearn.manifold import TSNE
import numpy as np
from  matplotlib import pyplot as plt

model2 = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
test_ds = np.concatenate(list(train_ds.take(5).map(lambda x, y : x))) # get five batches of images and convert to numpy array
features = model2(test_ds)
labels = np.argmax(model(test_ds), axis=-1)
tsne = TSNE(n_components=2).fit_transform(features)

def scale_to_01_range(x):

    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range

tx = tsne[:, 0]
ty = tsne[:, 1]

tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)

# colors = ['red', 'blue', 'green', 'brown', 'yellow']
# classes = train_ds.class_names
# print(classes)
fig = plt.figure()
ax = fig.add_subplot(111)
# for idx, c in enumerate(colors):
#     indices = [i for i, l in enumerate(labels) if idx == l]
#     current_tx = np.take(tx, indices)
#     current_ty = np.take(ty, indices)
#     ax.scatter(current_tx, current_ty, c=c, label=classes[idx])

ax.scatter(tx, ty)

ax.legend(loc='best')
plt.show()