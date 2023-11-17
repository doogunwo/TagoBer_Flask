import numpy as np
import face_recognition
import os
import cv2
from PIL import Image


def faceList(folder_path):

    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    return subfolders

def name2():
    path = "Dataset"
    lit = faceList(path)


    known_face_name = []

    for i in lit:
        known_face_name.append(i.split("/")[1])
    
    return known_face_name


def encoding():
    path = "Dataset"
    lit = faceList(path)

    known_face_encodings = []

    for i in lit:
        image_files = [f for f in os.listdir(i) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        
        for image_file in image_files:
            image_path = os.path.join(i, image_file)
            
            try:
                img = face_recognition.load_image_file(image_path)
                face_encoding = face_recognition.face_encodings(img)[0]
                known_face_encodings.append(face_encoding)
            except Exception as e:
                print(f"An error occurred while processing {image_path}: {e}")

    return known_face_encodings



def run():

    known_face_encodings = encoding()
    known_face_names = name2()
    

    video_capture = cv2.VideoCapture(0)

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame =True
    while True:
        ret, frame = video_capture.read()

        if not ret:
            print("Error: Failed to read frame from video stream.")

        # 적절한 처리 추가
        if process_this_frame:
        
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            rgb_small_frame = small_frame[:, :, ::-1]


            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Display the resulting image
        cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
        

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ =="__main__":
    run()






