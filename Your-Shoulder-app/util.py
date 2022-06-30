import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from os.path import exists as file_exists

class face:
    def __init__(self, img):
        self.img = img
        
    def cropDetect(self):
        '''
        ------ DETECT AND CROP ------
        This function detects the face and return the gary scaled image
        with face croped.
        
        File Needed: Image, haarcascade_frontalface_alt.xml
        Returns 
        
        '''
        
        # Read the haarcascade_frontalface_alt file
        if file_exists('static/cnn_model_utils/haarcascade_frontalface_alt.xml'):
            face_cascade = cv2.CascadeClassifier('static/cnn_model_utils/haarcascade_frontalface_alt.xml')
        else:
            raise Exception("Sorry, haarcascade_frontalface_alt file not found.. Download the file from https://github.com/opencv/opencv/tree/master/data/haarcascades")
        
        # read the image
        image = np.array(Image.open(self.img))
        
        # Convert the image to gray scale
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Using the haarcascade_frontalface_alt file to detect the face
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.3, minNeighbors = 5)
        
        # Draw a rectangle around an image
        for x,y,w,h in faces:
            image = cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0), 2)
            
        # plt.imshow(image) #show the image with a detected face
        
        # crop the image
        for x,y,w,h in faces:
            extracted_img = image[y:y+h, x:x+w]
            resized_img = cv2.resize(extracted_img,(int(extracted_img.shape[1]/2), int(extracted_img.shape[0]/2)))
            gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            
        # cv2.imshow('', extracted_img)
        # cv2.imshow('', resized_img)
        # cv2.imshow('', gray_img)
        # plt.imshow(gray_img);
        
        return gray_img



def trigger(text):
    ls = ['happy', 'jovial', 'joyful', 'love', 'like', 'content', 'glad', 'positive', 'cheery', 'active', 'energized', 'passionate', 'elated', 'excited', 'enthusiastic', 'lucky', 'yoshi', 'garvit', 'sonakshi', 'aabha', 'dhriti', 'hy', 'hello', 'hi']
    dep = ['discourage', 'demotivated', 'sad', 'stubborn', 'not happy', 'pessimistic', 'miserable', 'suffer', 'pain', 'painful', 'grieving', 'harm', 'hurt', 'hurtful', 'unhappy', 'upset', 'low', 'regretful', 'gloomy', 'broken', 'demoralized', 'heartsick']
    for _ in ls:
        if _ in text.lower():
            return [1, 0, 0]
    for _ in dep:
        if _ in text.lower():
            return [0, 0, 1]
    return 0



        
        
