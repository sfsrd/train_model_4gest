from keras.models import load_model
from keras.models import model_from_yaml
import cv2
import numpy as np

yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)
model.load_weights("model.h5")
print("Model is loaded")

imres=cv2.imread('frame_07_07_0040.png',0)
#imres=cv2.imread('cropped.jpg',0)
imres = cv2.resize(imres, (320,120))
print("Image is loaded")
cv2.imshow('image',imres)
#im=image[y1:y2,x1:x2] #crop hand from object detection part
#imres=cv2.resize(imres, (320,120))  #resize image to image size in dataset
imres = imres.astype('float32')
imres = imres.reshape((1, 120, 320, 1))
imres /= 255
predicted = model.predict_on_batch(imres)
lable = np.argmax(predicted,axis=1)
print('lable=', lable)

