# -*- coding: utf-8 -*-
"""
@author: karthik
"""

# import the libraries
import os
import face_recognition

# make a list of all the available images
images = os.listdir('images')

# load your image
image_to_be_matched = face_recognition.load_image_file('NarendraModi/NM11.jpg')

# encoded the loaded image into a feature vector
image_to_be_matched_encoded = face_recognition.face_encodings(
    image_to_be_matched)[0]

known_faces = []
# iterate over each image
for image in images:
    # load the image
    current_image = face_recognition.load_image_file("images/" + image)
    # encode the loaded image into a feature vector
    current_image_encoded = face_recognition.face_encodings(current_image)[0]
    # add encoded face to list
    known_faces.append(current_image_encoded)


# match your image with the image and check if it matches
result = face_recognition.compare_faces(
    known_faces, image_to_be_matched_encoded, tolerance=0.6)

print(result)
