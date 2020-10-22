import warnings
warnings.filterwarnings('ignore')

import cv2 as cv2
import face_recognition

# this is the image we are checking
elon_image = face_recognition.load_image_file('C:/Users/Akhil/Desktop/images_face/elon_musk.jpg')
elon_image = cv2.cvtColor(elon_image,cv2.COLOR_BGR2RGB)

# this is the test image which we are using for testing
elon_test = face_recognition.load_image_file('C:/Users/Akhil/Desktop/images_face/bill_gates.jpg')
elon_test = cv2.cvtColor(elon_test,cv2.COLOR_BGR2RGB)

# finding the faces in the mage and finding the encoding the image
faceloc = face_recognition.face_locations(elon_image)[0]  #sending a single image we are getting the single element
encode_elon = face_recognition.face_encodings(elon_image)[0]
cv2.rectangle(elon_image,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),3)

# now applying in the test image we do not need the face_location here only need the face encoding
faceloc_test = face_recognition.face_locations(elon_test)[0]
encode_elon_test = face_recognition.face_encodings(elon_test)[0]
cv2.rectangle(elon_test,(faceloc_test[3],faceloc_test[0]),(faceloc_test[1],faceloc_test[2]),(255,0,255),3)

# we are comparing both the images

results = face_recognition.compare_faces([encode_elon],encode_elon_test)
facedist = face_recognition.face_distance([encode_elon],encode_elon_test)
cv2.putText(elon_test,f'{results} {round(facedist[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
print(results,facedist)

cv2.imshow('elon_musk',elon_image)
cv2.imshow('test_elonmusk',elon_test)
cv2.waitKey(0)
cv2.destroyAllWindows()


