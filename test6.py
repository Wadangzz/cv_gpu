import cv2
import socket
import time
from collections import Counter, deque
from ultralytics import YOLO

HOST = '127.0.0.1'
PORT = 6000

model = YOLO('./runs/detect/project2/weights/best.pt')
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

# 최근 프레임 결과 저장용 (ex: 최근 10개 저장)
# frame_buffer = deque(maxlen=10)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.GaussianBlur(frame, (5, 5), 0)
    sharpening = cv2.addWeighted(frame, 1.5, img, -0.5, 0)
    results = model.predict(sharpening, conf=0.6, iou=0.5)
    boxes = results[0].boxes

    if boxes.cls.numel() > 0:
    #     for cls in boxes.cls:
    #         frame_buffer.append(int(cls.item()))
    #     print("Frame Buffer:", list(frame_buffer))  # 버퍼 상태 확인

    # if len(frame_buffer) == frame_buffer.maxlen:
    #     count = Counter(frame_buffer)
    #     most_common_cls, freq = count.most_common(1)[0]
    #     print(f"Most Common Class: {most_common_cls}, Frequency: {freq}")

    #     if most_common_cls == 0:
            
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(b'2001,1')
                # print(f'[다수결] 클래스 0 선택 (빈도 {freq} / {frame_buffer.maxlen}) → 전송: 0,100')
            
        # else:
            
        #     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        #         s.connect((HOST, PORT))
        #         s.sendall(b'0,0')
        #         print(f'[다수결] 클래스 {most_common_cls} 선택 (빈도 {freq} / {frame_buffer.maxlen}) → 전송: 0,0')

        #     frame_buffer.clear()

    cv2.imshow("YOLOv8 Detection", results[0].plot())

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
