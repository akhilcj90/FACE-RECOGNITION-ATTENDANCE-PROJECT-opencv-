
import warnings
warnings.filterwarnings('ignore')

import cv2 as cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path = 'C:/Users/Akhil/Desktop/images_face'
images = []
classnames = []

mylist = os.listdir(path)
print(mylist)
for cl in mylist:
    curlimage = cv2.imread(f'{path}/{cl}')
    images.append(curlimage)
    classnames.append(os.path.splitext(cl)[0])
print(classnames)


def finencoding(images):
    encodinglist = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodinglist.append(encode)
    return encodinglist

def makeattendance(names):
    with open('attendance_list.csv',mode = 'r+')  as f:
        mydatalist = f.readlines()
        name_list = []
        for line in mydatalist:
            entry = line.split(',')
            name_list.append(entry[0])

        if names not in name_list:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{names},{dtString}')


encode_face_known = finencoding(images)
print('encoding completed')

cap = cv2.VideoCapture(0)

while True:
    rest,frame = cap.read()
    imgs  = cv2.resize(frame,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)

    # faces in the current frame 3 here there are a lot of people/faces in the frame so we are
    # passing the face_current_frame to the encoding
    face_current_frame = face_recognition.face_locations(imgs)
    encode_current_frame = face_recognition.face_encodings(imgs,face_current_frame)

    for encodeface,facelock in zip(encode_current_frame,face_current_frame):
        matches = face_recognition.compare_faces(encode_face_known,encodeface)
        dist = face_recognition.face_distance(encode_face_known,encodeface)
        print(dist)

        match_index = np.argmin(dist)

        if matches[match_index]:
            names = classnames[match_index].upper()
            print(names)
            y1,x2,y2,x1 = facelock
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, names, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            makeattendance(names)

cv2.imshow('webcam',frame)
cv2.waitKey(0)

