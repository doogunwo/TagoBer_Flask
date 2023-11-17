import face_recognition

# Load the jpg files into numpy arrays
biden_image = face_recognition.load_image_file("Dataset/dgw0601/dgw0601_og.jpg")
biden_image2 = face_recognition.load_image_file("IMG_2060.jpg")
unknown_image = face_recognition.load_image_file("IMG_2055.jpg")

# Get the face encodings for each face in each image file
# Since there could be more than one face in each image, it returns a list of encodings.
# But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
try:
    biden_face_encoding2 = face_recognition.face_encodings(biden_image2)[0]
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
   
except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()

known_faces = [
    biden_face_encoding,
    biden_face_encoding2
   
]

# results is an array of True/False telling if the unknown face matched anyone in the known_faces array
results = face_recognition.compare_faces(known_faces, unknown_face_encoding)

known_names = ["dgw0601","jhj"]  # Replace this with your actual list of names
print("Name: " + known_names[results.index(True)])