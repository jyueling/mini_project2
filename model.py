import tensorflow as tf
from tensorflow import keras
from sklearn.cross_validation import train_test_split
import random

import numpy as np
import os
import cv2

path_name = './'

images = []
labels = []
#read images from folders
def read_path(path_name):
    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
            
        if os.path.isdir(full_path):
            read_path(full_path)
        else:
            if dir_item.endswith('.JPG')or dir_item.endswith('.jpg') :
                image = cv2.imread(full_path)
                image = cv2.resize(image, (28, 28))
                    
                images.append(image)                
                labels.append(path_name)  
    return images, labels                              
#load images
images,labels = read_path(path_name)                  
images = np.array(images)
sizes=np.size(labels)
print ("labels")
labels2=np.zeros((sizes,1))

#label images
j=0
for i in labels:
    if i.endswith('sunflower'):
        labels2[j]=0
        j+=1
    else:
        labels2[j]=1
        j+=1
labels = labels2



#class_names = ['sunflower','rose']

#divide images into train image and test image
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size = 0.3, random_state = random.randint(0, 100))        
 
#reshape images
train_images = train_images.reshape(train_images.shape[0], 3, 28, 28)
test_images = test_images.reshape(test_images.shape[0], 3, 28, 28)


train_images = train_images / 255.0

test_images = test_images / 255.0

#model
model = keras.Sequential([
    keras.layers.Convolution2D(32, 3, 3,input_shape = (3,28,28)),
    keras.layers.Activation('relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(2, activation=tf.nn.softmax),
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)
#save model
model.save('./projectmodel.h5')
#test model accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

#predictions = model.predict(test_images)
#print(predictions)
