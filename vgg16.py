# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 01:15:28 2021

@author: Fahimul
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dense
from keras.layers import Flatten
from keras import optimizers

classifier = Sequential()

# Please note the following code follows the architecure of VGG16 model;
# however, in instead of 224*224 RGB images in traditional VGG16, the model
# has been fed with 64*64 HSV color space images
classifier.add(Conv2D(input_shape=(64,64,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
classifier.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
classifier.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
classifier.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
classifier.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
classifier.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
classifier.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
classifier.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
classifier.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
classifier.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
classifier.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
classifier.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
classifier.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
classifier.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
classifier.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
classifier.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
classifier.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
classifier.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(units=4096,activation="relu"))
classifier.add(Dense(units=4096,activation="relu"))
classifier.add(Dense(units=36, activation="softmax"))

#Compiling The Designed VGG16 Model, learning rate = 0.00001
classifier.compile(
              optimizer = optimizers.Adam(lr = 0.00001),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

#Part 2 Fittting the model to the dataset
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'C:/Users/fahim/Desktop/ENEL 645/Final Project/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

validation_set = test_datagen.flow_from_directory(
        'C:/Users/fahim/Desktop/ENEL 645/Final Project/validation_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

'''#Saving the model import h5py'''
classifier.save('C:/Users/fahim/Desktop/ENEL 645/Final Project/Handgesture_Final_VGG.h5')

# Defining callabcks
from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("Handgesture_Final_VGG.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')

model = classifier.fit_generator(
        training_set,
        steps_per_epoch=64,
        epochs=50,
        verbose=1,
        validation_data= validation_set,
        validation_steps=48)

print(classifier.summary())

print(model.history.keys())
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(model.history['accuracy'], label = "Train accuarcy")
plt.plot(model.history['val_accuracy'],label = "Val accuarcy")
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss

plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# "Accuracy from test sets"
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'C:/Users/fahim/Desktop/ENEL 645/Final Project/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')
loss,accuracy = classifier.evaluate_generator(test_set, steps=56)
print("Testing accuracy:",accuracy)
