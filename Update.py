import os

import cv2
import mysql.connector

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='1234',
    database='sys'
)
cursor = conn.cursor()
query = "select username, imagePath from sys.faceregister"
#select username, imagePath from sys.faceregister where username= 'dgw0601'
cursor.execute(query)
root = 'Dataset'
res = cursor.fetchall()
for row in res:
    try:

        os.mkdir(os.path.join(root, row[0]))
        if row[1] != "":
            img_og = cv2.imread(row[1])
            img_cut = img_og.copy()
            faces = face_cascade.detectMultiScale(img_cut, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                face = img_cut[y:y + h, x:x + w]
                cv2.imwrite(os.path.join(root, row[0])+"/"+row[0]+".jpg", face)
                print("1")

            cv2.imwrite(os.path.join(root, row[0]) + "/" + row[0] + "_og.jpg", img_og)


        else:
            pass
            print("hi")
    except FileExistsError:
        print("ex")
        pass

cursor.close()
conn.close()