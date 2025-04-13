import MyModule.myhandlandmark as ml
import MyModule.object as object
import cv2
import socket
import time
from ultralytics import YOLO
from ultralytics.engine.results import Results


HOST = '127.0.0.1'
PORT = 6000

model = YOLO('./MyModule/project.pt')
# hand = ml.handtracking()
# obj = object.Object()

# 비디오 파일 로드
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))  # 루프 바깥에서 딱 1번 연결

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        

        height, width, _ = frame.shape
        img = cv2.GaussianBlur(frame, (5, 5), 0)

        sharpening = cv2.addWeighted(frame, 1.5, img, -0.5, 0)
        results = model.predict(sharpening, conf = 0.5, iou = 0.5)
        boxes = results[0].boxes
        print(f'{boxes.cls},{boxes.conf}')

        trigger = 0  # default는 OFF

        for cls, conf in zip(boxes.cls, boxes.conf):
            if int(cls.item()) == 1:
                trigger = 1

            # 트리거 값 TCP 전송
            if trigger == 1:
                s.sendall(b'100,100')
            else:
                s.sendall(b'100,0')


        # 화면 출력
        cv2.imshow("YOLOv8 Detection", results[0].plot())

        if cv2.waitKey(delay) & 0xFF == ord('q'):  # 'q' 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()
