#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np
import requests
from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
from pydub import AudioSegment
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
import os
import gc
import soundfile as sf
from scipy.io.wavfile import write
import itertools
import tensorflow
import keras
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from tensorflow.keras.models import Sequential 
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dropout, Flatten, Dense  
from tensorflow.keras import applications  
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import math  
import datetime
import time
from time import sleep


# In[15]:


N_FFT = 1024         
HOP_SIZE = 1024       
N_MELS = 128            
WIN_SIZE = 1024      
WINDOW_TYPE = 'hann' 
FEATURE = 'mel'      
FMIN = 1400 


# In[20]:


#Loading vgg16 model
vgg16 = applications.VGG16(include_top=False, weights='imagenet')


# # Load saved model

# In[2]:


model = keras.models.load_model('D:/C Drive Documents/Bird_Sound_Recognition/My_Model')


# # Testing on new images

# In[10]:


def removeSilence(signal):
    return signal[librosa.effects.split(signal)[0][0] : librosa.effects.split(signal)[0][-1]]


# In[9]:


def mel_spectogram_generator(audio_name,signal,sample_rate,augmentation,target_path):
    S = librosa.feature.melspectrogram(y=signal,sr=sample_rate,
                                    n_fft=N_FFT,
                                    hop_length=HOP_SIZE, 
                                    n_mels=N_MELS, 
                                    htk=True, 
                                    fmin=FMIN, 
                                    fmax=sample_rate/2) 

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S**2,ref=np.max), fmin=FMIN,y_axis='linear')
    plt.axis('off')
    plt.savefig(target_path + augmentation + audio_name[:-4] + '.png',bbox_inches='tight',transparent=True, pad_inches=0)
    plt.clf()
    plt.close("all")
    gc.collect()
    


# In[11]:


def read_image(file_path):
    print("[INFO] loading and preprocessing image...")  
    image = load_img(file_path, target_size=(558, 217))  
    image = img_to_array(image)  
    image = np.expand_dims(image, axis=0)
    image /= 255.  
    return image


# In[23]:


def test_single_image(path):
    birds = ['AshyPrinia',
 'AsianKoel',
 'BlackDrongo',
 'CommonMyna',
 'CommonTailorbird',
 'GreaterCoucal',
 'GreenBee-eater',
 'IndianRobin',
 'LaughingDove',
 'White-throatedKingfisher']
    images = read_image(path)
    time.sleep(.5)
    bt_prediction = vgg16.predict(images)  
    preds = model.predict_proba(bt_prediction)
    for idx, bird, x in zip(range(0,10), birds , preds[0]):
        print("ID: {}, Label: {} {}%".format(idx, bird, round(x*100,2) ))
    print('Final Decision:')
    time.sleep(.5)
    for x in range(3):
        print('.'*(x+1))
        time.sleep(.2)
    class_predicted = model.predict_classes(bt_prediction)
    for idx, bird, x in zip(range(0,10), birds , preds[0]):
        if idx == class_predicted[0]:
            print("ID: {}, Label: {}".format(class_predicted[0], bird))
    return load_img(path)


# In[29]:


def predict_bird_sound(source_path,file_name, target_path = 'D:/'):
    N_FFT = 1024         
    HOP_SIZE = 1024       
    N_MELS = 128            
    WIN_SIZE = 1024      
    WINDOW_TYPE = 'hann' 
    FEATURE = 'mel'      
    FMIN = 1400
    augmentation = ''

    signal, sample_rate = librosa.load(source_path +  file_name,sr = None)
    DNsignal = removeSilence(signal)
    mel_spectogram_generator(file_name,DNsignal,sample_rate,'',target_path)
    path =  target_path  + augmentation + file_name[:-4] + '.png'
    test_single_image(path)


# In[37]:


print("BIRD SOUND RECOGNITION APP - By Karthik Mandapaka")
sleep(1)
print("Welcome")
sleep(2)
while(1):
    source_path = input("Please enter Source path:  ")
    sleep(2)
    file_name = input("Please enter the audio file name:  ")
    sleep(2)
    print("Recognizing bird sound")
    sleep(0.5)
    print('.')
    sleep(0.5)
    print('..')
    sleep(0.5)
    print('...')
    predict_bird_sound(source_path,file_name)
    cont = input("Do you want to identify another bird sound?(Enter 1 for Yes or 0 for No)")
    if (cont == '0'): break


# In[25]:


# predict_bird_sound('D:/C Drive Documents/Bird_Sound_Recognition/Data for each bird/data/xeno-canto-dataset/LaughingDove/','Spilopelia280683.wav','D:/')

