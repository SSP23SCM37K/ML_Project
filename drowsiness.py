import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

mixer.init()
snd = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
lefteye = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
righteye = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')



label=['Close','Open']

model = load_model('cnnmdl1.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thc=2
rpredictions=[99]
lpredictions=[99]

while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = lefteye.detectMultiScale(gray)
    right_eye =  righteye.detectMultiScale(gray) 


    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpredictions = np.argmax(model.predict(r_eye), axis=-1) 
        # print(rpredictions)
        if (rpredictions.all()==1):
            label='Open' 
        if (rpredictions.all()==0):
            label='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpredictions = np.argmax(model.predict(l_eye), axis=-1) 
        if (lpredictions.all()==1):
            label='Open'   
        if (lpredictions.all()==0):
            label='Closed'
        break

    if np.logical_and((rpredictions.all()) == 0, (lpredictions.all()) == 0):
        score=score+1
        
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpredictions[0]==1 or lpredictions[0]==1):
    else:
        score=score-1
        
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>7):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            snd.play()
            
        except:  # isplaying = False
            pass
        if(thc<16):
            thc= thc+2
        else:
            thc=thc-2
            if(thc<2):
                thc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thc) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
