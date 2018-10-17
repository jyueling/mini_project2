import cv2
import numpy as np
from keras.models import load_model




if __name__ == '__main__':

    
    classname = ['sunflower','rose']
    model = load_model('./projectmodel.h5')
    image = cv2.imread('2.jpg')
    image = cv2.resize(image, (28, 28))
    image = image.reshape(1, 3, 28, 28)
    image = image / 255.0
    result = model.predict(image)
    
    if np.argmax(result) == 0:
        print('sunflower')
    else:
        print('rose')
