# Part 1 - Building the CNN
#importing the Keras libraries and packages

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras import optimizers

# Initialing the CNN
classifier = Sequential()


# Step 1 - Convolution Layer
classifier.add(Convolution2D(32, (3,  3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(Convolution2D(32, (3,  3), input_shape = (64, 64, 3), activation = 'relu'))
#step 2 - Pooling
classifier.add(MaxPooling2D(pool_size =(2,2)))

# Adding second convolution layer
classifier.add(Convolution2D(32, (3,  3), padding= 'same',  activation = 'relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))

#Adding 3rd Convolution Layer
classifier.add(Convolution2D(64, (3,  3), padding= 'same',  activation = 'relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))


#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full Connection
classifier.add(Dense(784, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(256, activation = 'relu'))
classifier.add(Dense(36, activation = 'softmax'))

#Compiling The CNN
classifier.compile(
              optimizer = optimizers.Adam(lr = 0.00001),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

#Part 2 Fittting the CNN to the image
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'C:/Users/fahim/Desktop/ENEL 645/Final Project/training_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

validation_set = test_datagen.flow_from_directory(
        'C:/Users/fahim/Desktop/ENEL 645/Final Project/validation_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')



model = classifier.fit_generator(
        training_set,
        steps_per_epoch=168,
        epochs=50,
        verbose=1,
        validation_data= validation_set,
        validation_steps=56)

print(classifier.summary())
'''#Saving the model
import h5py'''
classifier.save('C:/Users/fahim/Desktop/ENEL 645/Final Project/Handgesture_Final_CNN_d.h5')

print(model.history.keys())
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(model.history['accuracy'], label = "Train accuarcy")
plt.plot(model.history['val_accuracy'],label = "Val accuarcy")
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
# summarize history for loss

plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
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







