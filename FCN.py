"""
Created on Sat Apr 17 01:15:28 2021

@author: Fahimul
"""
# Part 1 - Building the FCN
#importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras import optimizers
import tensorflow as tf
# Initialing the FCN
classifier = Sequential()

# Step 1 - Creating model input
classifier.add(tf.keras.Input(shape=(64, 64, 3)))

#Step 2 - Flattening
classifier.add(Flatten())

#Step 4 - Full Connection

classifier.add(Dense(512, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(512, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(256, activation = 'relu'))
#classifier.add(Dropout(0.3))
classifier.add(Dense(36, activation = 'softmax'))

#Compiling The Designed FCNN model
classifier.compile(
              optimizer = optimizers.Adam(lr = 0.00001),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

#Part 2 Fittting the FCNN to the dataset
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2)

val_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'C:/Users/fahim/Desktop/ENEL 645/Final Project/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

validation_set = val_datagen.flow_from_directory(
        'C:/Users/fahim/Desktop/ENEL 645/Final Project/validation_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

model = classifier.fit_generator(
        training_set,
        steps_per_epoch=64,
        epochs=50,
        verbose=1,
        validation_data= validation_set,
        validation_steps= 48)

'''#Saving the model
import h5py'''
classifier.save('C:/Users/fahim/Desktop/ENEL 645/Final Project/Handgesture_Final_FCN.h5')

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

