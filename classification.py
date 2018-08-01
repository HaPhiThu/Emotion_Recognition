from __future__ import print_function
import numpy as np
import cv2
from PIL import Image
import os, sys
from time import sleep

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import cnn_major


#label matrix:
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# The directory contains the input image
data_dir="/home/ha/Desktop/Facial-Expression-Recognition-master/test_set"

#load pretrained-model saved in model4layer_2_2_pool.h5 file.
json_file = open('/home/ha/Desktop/Facial-Expression-Recognition-master/model_4layer_2_2_pool.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#load weight into above model.
loaded_model.load_weights("/home/ha/Desktop/Facial-Expression-Recognition-master/model_4layer_2_2_pool.h5")
print("Loaded model from disk")
while 1: 
    #check tes_set is empty or not to normalizing after forwarding into pre-trained model :
    if(os.path.isdir(data_dir)):
        if (os.listdir(data_dir)!=[]):
            #get the input image
            image_name = os.listdir(data_dir)
            image_path=os.path.join(data_dir,*image_name)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            print("image size:",np.shape(img))
            
            #resize the input image
            img=cv2.resize(img, (48,48))
            print("image size:",np.shape(img))
            
            #convert gray image into pixel matrix
            WIDTH=48
            HEIGHT = 48
            data = img.reshape(1,48,48,1)
    
            # Model will predict the probability values for 7 labels for a test image
            score = loaded_model.predict(data)
            print (score)

            #score1=np.array(score)
            label_index=np.argmax(score)
            print (label_map[label_index])
            
            #remove the input image after predicting the probability
            filelist = [ f for f in os.listdir("/home/ha/Desktop/Facial-Expression-Recognition-master/test_set")]
            for f in filelist:
                os.remove(os.path.join("/home/ha/Desktop/Facial-Expression-Recognition-master/test_set", f))
        else: 
            sleep(1)
            print("Please wait")









