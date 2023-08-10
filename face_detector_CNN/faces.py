import time
import tensorflow as tf
import os
import numpy as np
from keras import layers
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt


def makediscriminator():
    in_shape = (32, 32, 3)
    model = tf.keras.Sequential()

    # normal
    model.add(layers.Conv2D(64, (3, 3),
                            padding='same',
                            input_shape=in_shape))
    model.add(layers.LeakyReLU(alpha=0.2))

    # down
    model.add(layers.Conv2D(128, (3, 3),
                            padding='same',
                            input_shape=in_shape))
    model.add(layers.LeakyReLU(alpha=0.2))

    # down
    model.add(layers.Conv2D(32, (3, 3),
                            padding='same',
                            input_shape=in_shape))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='tanh'))

    return model


def makegenerator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(256 * 4 * 4, use_bias=False, input_shape=(100,)))
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((4, 4, 256)))

    # Upsample to 8x8
    model.add(layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))

    # Upsample to 16x16
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))

    # Upsample to 32x32
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(3, (3, 3), activation='tanh', padding='same'))

    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradientsgen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradientsdisc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradientsgen, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradientsdisc, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        y = 0
        for image_batch in dataset:
            print(y+1)
            y = y + 1
            train_step(image_batch)

        print("Time for epoch {} is {} sec".format(epoch+1, time.time() - start))


path = "./train"
files = os.listdir(path)

BATCH_SIZE = 256
noise_dim = 100

image_paths = []
ones_labels = []

# Data Wrangling
for filename in files:
    image_path = os.path.join(path, filename)
    split = filename.split("_")
    if len(split) == 4:
        image_paths.append(image_path)
        ones_labels.append(1)

features = []

for image in image_paths:
    img = load_img(image)
    img = img.resize((32, 32))
    img = tf.cast(img, tf.float64)
    img = (img - 127.5) / 127.5
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    features.append(img)

X = np.array(features)

noise = tf.random.normal([1, 100])
generator = makegenerator()
discriminator = makediscriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

noise_dim = 100

noise = tf.random.normal([1, 100])
before = generator(noise, training=False)
train(X, 2)

savefile = "./GAN/generator"
generator.save(savefile)

after = generator(noise, training=False)

plt.subplot(1, 2, 1)
plt.imshow(before[0, :, :, 0])

plt.subplot(1, 2, 2)
plt.imshow(after[0, :, :, 0])
plt.show()

