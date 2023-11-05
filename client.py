import pickle
import time
import cv2
import requests

def start():
    url  ="http://14.49.83.210:5000/start"
    #http://14.42.240.116:82/start
    #처음 실행했을 때 모델 로드하도록하자
    response = requests.post(url)
    if response.status_code == 200:
        res = pickle.loads(response.content)
        print(res)

def streaming():
    if not camera.isOpened():
        print("카메라를 열 수 없습니다.")
        exit()

    url = "http://14.49.83.210:5000/video_feed"
    try:
        while True:
            print(time.time())#로그용
            ret, img = camera.read()
        # 프레임 이미지를 서버로 전송
            _, img_encoded = cv2.imencode('.jpg', img)
            response = requests.post(url, data=pickle.dumps(img_encoded.tobytes()))

# 서버에서 전송한 탐지 결과 출력
            if response.status_code == 200:
                result = pickle.loads(response.content)
                print(result)#로그용
    except ConnectionError:
        print(time.time())
        pass

if __name__ == "__main__":
    camera = cv2.VideoCapture(0)
    start()
    streaming()

