import pickle
from os import listdir
from os.path import isdir, isfile, join

import mtcnn
import numpy as np
from flask import Flask, request, Response

detector = mtcnn.MTCNN()
import mysql.connector
import os
import cv2
import time

app = Flask(__name__)
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='1234',
    database='sys'
)


def train(name):
    data_path = 'Dataset/' + name + '/'
    # 파일만 리스트로 만듬
    face_pics = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    Training_Data, Labels = [], []

    for i, files in enumerate(face_pics):
        image_path = data_path + face_pics[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # 이미지가 아니면 패스
        if images is None:
            continue
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)
    if len(Labels) == 0:
        print("There is no data to train.")
        return None
    Labels = np.asarray(Labels, dtype=np.int32)
    # 모델 생성
    model = cv2.face.LBPHFaceRecognizer_create()
    # 학습
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    print(name + " : Model Training Complete!!!!!")

    # 학습 모델 리턴
    return model

def trains():
    # faces 폴더의 하위 폴더를 학습
    data_path = 'Dataset/'
    # 폴더만 색출
    model_dirs = [f for f in listdir(data_path) if isdir(join(data_path, f))]

    # 학습 모델 저장할 딕셔너리
    models = {}
    # 각 폴더에 있는 얼굴들 학습
    for model in model_dirs:
        print('model :' + model)
        # 학습 시작
        result = train(model)
        # 학습이 안되었다면 패스!
        if result is None:
            continue
        # 학습되었으면 저장
        print('model2 :' + model)
        models[model] = result

    # 학습된 모델 딕셔너리 리턴
    return models
def face_detector(img, size=0.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if faces is ():
            return img, []
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi = img[y:y + h, x:x + w]
            roi = cv2.resize(roi, (200, 200))
        return img, roi  # 검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라(roi) 전

def run(models,frame):
    # 카메라 열기

        # 얼굴 검출 시도
        image, face = face_detector(frame)
        try:
            min_score = 999  # 가장 낮은 점수로 예측된 사람의 점수
            min_score_name = ""  # 가장 높은 점수로 예측된 사람의 이름

            # 검출된 사진을 흑백으로 변환
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # 위에서 학습한 모델로 예측시도
            for key, model in models.items():
                result = model.predict(face)

                if min_score > result[1]:
                    min_score = result[1]
                    min_score_name = key

        except:
            pass

        return min_score_name

@app.route('/start', methods=['POST'])
def start():
    res = "client connect"
    # 결과 전송
    response = {'result': res}
    return Response(response=pickle.dumps(response), status=200, mimetype='application/octet-stream')

@app.route('/Updata', methods=['POST'])
def update():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cursor = conn.cursor()
    query = "select username, imagePath from sys.faceregister"
    # select username, imagePath from sys.faceregister where username= 'dgw0601'
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
                    cv2.imwrite(os.path.join(root, row[0]) + "/" + row[0] + ".jpg", face)
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


@app.route('/video_feed', methods=['POST'])
def video_feed():
        start_time = time.time()
        passenger = []
        while(1):

            time.sleep(0.3)
            img_encoded = pickle.loads(request.data)
            img = cv2.imdecode(np.frombuffer(img_encoded, np.uint8), -1)
            # 얼굴 검출 수행
            res=run(models,img)
            current_time = time.time()

            print(res)

            if res not in passenger:
                if res == '':
                    pass
                else:
                    if start_time - current_time>4.0:
                        start_time = current_time
                        passenger.append(res)
                    else:
                        pass



            print(passenger)
            response = {'result': res}
            return Response(response=pickle.dumps(response), status=200, mimetype='application/octet-stream')

@app.route('/send_log',methods=['POST'])
def sendLog():
    print("nodejs와 연결")


if __name__ == '__main__':

    models = trains()
    app.run(host='192.168.1.192', port=5000, debug=True)