import tensorflow as tf
import os
from keras.preprocessing.image import load_img
import numpy as np
import matplotlib.pyplot as plt

global gendermodel
global agemodel
global raceModel

gendermodel = tf.keras.models.load_model('./CNN/gender')
agemodel = tf.keras.models.load_model('./CNN/Age')
racemodel = tf.keras.models.load_model('./CNN/race')

def predict(title, img):

    gender = gendermodel.predict(img)
    if gender < 0.5:
        genderlabel = "Male"
    else:
        genderlabel = "Female"

    age = agemodel.predict(img)
    agelabel = round(age[0][0])

    race = racemodel.predict(img)
    indice = np.where(race == np.amax(race))
    racenum = indice[1][0]
    if racenum == 0:
        racelabel = "White"
    if racenum == 1:
        racelabel = "Black"
    if racenum == 2:
        racelabel = "Asian"
    if racenum == 3:
        racelabel = "Indian"
    if racenum == 4:
        racelabel = "Other"

    print(title)
    print("Age: " + str(agelabel))
    print("Gender: " + genderlabel)
    print("Race: " + racelabel)
    print()



path = "./test"
files = os.listdir(path)
x = 0

image_paths = []
age_labels = []

# Data Wrangling
for filename in files:
    image_path = os.path.join(path, filename)
    image_paths.append(image_path)

features = []

for image in image_paths:
    if image.endswith(".jpg"):
        img = load_img(image)
        img = img.resize((64, 64))
        img = tf.cast(img, tf.float32)
        img /= 255
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        predict(image, img)



