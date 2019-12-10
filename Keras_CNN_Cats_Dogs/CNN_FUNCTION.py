import os
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten, MaxPooling2D, Dropout
from keras.utils import to_categorical
from keras import backend as K
from keras.models import Model
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def create_model():
    #Init model
    model = Sequential()

    #1st Conv Layer
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=(100,100,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    #2nd Conv Layer
    model.add(Conv2D(32,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    #3rd Conv Layer
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #4th Conv Layer
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    #Flatten
    model.add(Flatten())

    #Full connection
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model

def get_train_data(path):
    #Get ImageDataGenerator Object in order to grab train images
    train_datagen = ImageDataGenerator(rescale=1./255)

    #Tell the ImageDataGenerator Object to get all images from a directory
    train_generator = train_datagen.flow_from_directory(
                        path,
                        target_size=(100,100),
                        batch_size = 50,
                        class_mode = 'binary')

    return train_generator



def get_test_data(path):
    #Get ImageDataGenerator Object in order to grab testimages
    test_datagen = ImageDataGenerator(rescale=1./255)

    #Tell the ImageDataGenerator Object to get all images from a directory
    test_generator = test_datagen.flow_from_directory(
                        path,
                        target_size=(100,100),
                        batch_size = 50,
                        class_mode = 'binary')

    return test_generator


def get_layer_info(layer_name,img_name,model):
    #Get intermediate layer output
    all_img = []
    img = cv2.imread(img_name)
    if img is not None:
        img = cv2.resize(img,(100,100),3)
        all_img.append(img)
    else:
        print("Image Not Loaded")

    X_test = np.array(all_img)



    intermediate_layer_model = K.function([model.layers[0].input],
                                          [model.layers[3].output])
    layer_output = intermediate_layer_model([X_test])[0]


    #Get subplot
    count = 0
    for i in range(layer_output.shape[-1]):
        plt.imshow(layer_output[0,:,:,i], cmap='gray')
        plt.savefig("layer" + str(count)+ '.png')
        count = count + 1
    plt.close()


#Predicition function
def predict_animal(path,model):
    print("Animal Predicition")
    img = cv2.imread(path)
    if img is not None:
        img = cv2.resize(img,(100,100))
        img = img.reshape(1,100,100,3)
        print(model.predict(img))
    else:
        print("Image Not Loaded")

def run_CNN():
    model = create_model()
    model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    #Give paths for Train/Test Images
    train_generator = get_train_data("{train dir}")
    validation_generator = get_test_data("{validation dir}")

    #Fit models
    model.fit_generator(train_generator,validation_data=validation_generator,epochs=30)

  

    get_layer_info("conv2d_2",'{Animal Pic}',model)
 


if __name__ == '__main__':
    run_CNN()
