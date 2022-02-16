# Importing the various libraries used
import cv2
import numpy as np
import face_recognition

# Step 1 of 3
# Loading KB image
imgKB = face_recognition.load_image_file('ImagesMain/KB.jpg')
# Converting KB image from BGR to RGB
imgKB = cv2.cvtColor(imgKB, cv2.COLOR_BGR2RGB)

# Loading KBTest image
imgKBTest = face_recognition.load_image_file('ImagesMain/JC.jpg')
# Converting KBTest image from BGR to RGB
imgKBTest = cv2.cvtColor(imgKBTest, cv2.COLOR_BGR2RGB)

# Step 2 of 3
# Locating the face in KB img ([0] since we are sending in only 1 image)
faceLocKB = face_recognition.face_locations(imgKB)[0]
# Encoding the face in KB img (measure similarity between the 2 images)
encodeKB = face_recognition.face_encodings(imgKB)[0]
# Creates rectangle around Kobe img (x1, y1, x2, y2 in rectangle, color purple rectangle, thickness of rectangle)
cv2.rectangle(imgKB, (faceLocKB[3], faceLocKB[0]), (faceLocKB[1], faceLocKB[2]), (255, 0, 255), 2)

# Locating the face in KBTest img ([0] since we are sending in only 1 image)
faceLocKBTest = face_recognition.face_locations(imgKBTest)[0]
# Encoding the face in KBTest img (measure similarity between the 2 images)
encodeKBTest = face_recognition.face_encodings(imgKBTest)[0]
# Creates rectangle around KobeTest img (x1, y1, x2, y2 in rectangle, color purple rectangle, thickness of rectangle)
cv2.rectangle(imgKBTest, (faceLocKBTest[3], faceLocKBTest[0]), (faceLocKBTest[1], faceLocKBTest[2]), (255, 0, 255), 2)

# Step 3 of 3
# Comparing the similarity of the faces in each image (outputs true or false)
results = face_recognition.compare_faces([encodeKB], encodeKBTest)
# Finding how similar these images are with the distances (lower the distance the better the match is)
faceDis = face_recognition.face_distance([encodeKB], encodeKBTest)
print(results, faceDis)
# Displaying the true or false value and face distance between the 2 images (on the test image)
# Results (true or false), element of [0] since its an array : rounding the distance to 2 decimals, origin, font of true or false value, scale, color red, thickness
cv2.putText(imgKBTest, f'{results} {round(faceDis[0],2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

# Naming KB image 'Kobe Bryant'
cv2.imshow('Kobe Bryant', imgKB)
# Naming KB Test image 'Rookie Kobe'
cv2.imshow('Rookie Kobe', imgKBTest)
# Displaying image for 0ms (keeps displaying the image until key is pressed to break)
cv2.waitKey(0)