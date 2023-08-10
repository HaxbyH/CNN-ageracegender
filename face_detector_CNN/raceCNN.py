import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt

plots = False

def create_plots(train, test, train_l, test_l, epoch_count):
    epoch_array = []
    for x in range(epoch_count):
        epoch_array.append(x + 1)
    plt.subplot(1, 2, 1)
    plt.plot(epoch_array, train, '-r', label="Training Accuracy")
    plt.plot(epoch_array, test, '-b', label="Testing Accuracy")
    plt.title("Training vs Testing Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epoch_array, train_l, '-r', label="Training Loss")
    plt.plot(epoch_array, test_l, '-b', label="Testing Loss")
    plt.title("Training vs Testing Loss")
    plt.legend()
    plt.show()


path = "./train"
files = os.listdir(path)

image_paths = []
age_labels = []
gender_labels = []
race_labels = []

# Data Wrangling
for filename in files:
    image_path = os.path.join(path, filename)
    split = filename.split("_")
    if len(split) == 4:
        image_paths.append(image_path)
        race_labels.append(int(split[2]))

features = []

for image in image_paths:
    img = load_img(image)
    img = img.resize((64, 64))
    img = tf.cast(img, tf.float32)
    img /= 255.0
    img = np.array(img)
    features.append(img)

X = np.array(features)
y = np.column_stack([race_labels])

# Creates Testing data 70/30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

epochs = 20

# Gender Model Creation
racemodel = tf.keras.models.Sequential()

racemodel.add(tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=(3, 3),
                                     activation='relu'))
racemodel.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                           strides=(1, 1)))

# Second
racemodel.add(tf.keras.layers.Conv2D(filters=16,
                                     kernel_size=(3, 3),
                                     activation='relu'))
racemodel.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                           strides=(1, 1)))

# Third
racemodel.add(tf.keras.layers.Conv2D(filters=8,
                                     kernel_size=(3, 3),
                                     activation='relu'))
racemodel.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                           strides=(1, 1)))

racemodel.add(tf.keras.layers.Flatten())
racemodel.add(tf.keras.layers.Dense(units=128, activation='relu'))
racemodel.add(tf.keras.layers.Dense(units=5, activation='softmax'))

racemodel.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

savefile = "./CNN/race"

callback = tf.keras.callbacks.ModelCheckpoint(filepath=savefile,
                                              save_weights_only=False,
                                              monitor="val_accuracy",
                                              mode="max",
                                              save_best_only=True)

train_info = racemodel.fit(X_train,
                           y_train,
                           epochs=epochs,
                           batch_size=64,
                           validation_split=0.2,
                           callbacks=[callback])

train_accuracy = train_info.history['accuracy']
test_accuracy = train_info.history['val_accuracy']
train_loss = train_info.history['loss']
test_loss = train_info.history['val_loss']

if plots:
    create_plots(train_accuracy, test_accuracy, train_loss, test_loss, epochs)

racemodel = tf.keras.models.load_model(savefile)

loss_train, accuracy_train = racemodel.evaluate(X_train, y_train, verbose=0)
loss_test, accuracy_test = racemodel.evaluate(X_test, y_test, verbose=0)

print("Training Accuracy %.2f" % accuracy_train)
print("Testing Accuracy %.2f" % accuracy_test)
