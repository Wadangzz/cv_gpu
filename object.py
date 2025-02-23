import onnxruntime as ort
import numpy as np
import cv2


boxes = []
confidences = []
class_ids = []

# COCO 클래스 로드
with open("coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ONNX YOLO 모델 로드
session = ort.InferenceSession("yolov8x.onnx", providers=["CUDAExecutionProvider"])

# 입력 텐서 정보 가져오기
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
output_names = [o.name for o in session.get_outputs()]

# print(f"Input Tensor Name : {input_name}")
# print(f"Input Tensor Shape : {input_shape}")
# print(f"Output Names : {output_names}")

# 비디오 파일 로드
cap = cv2.VideoCapture('highway.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 모델 입력 크기로 리사이징 
    size = 640 # 320 416 640 1280
    img = cv2.resize(frame, (size, size)) 
    img = img / 255.0  # 정규화
    img = img.transpose(2, 0, 1)
    # (H, W, C) → (C, H, W) 
    # frame의 attribute 순서를 RGB채널, 높이, 너비 순으로 변환
    img = np.expand_dims(img, axis=0).astype(np.float32)  # (1, C, H, W)
    # 1차원 추가해서 4차원 텐서 형태로 변환
    
    height, width, channels = frame.shape
    x_scale, y_scale = width/size, height/size

    # 모델 예측 실행
    outputs = session.run(output_names, {input_name: img})
    outs = outputs[0][0].T  # (84, 3549) → (3549, 84)
    # YOLOv3에 있는 4번째 index 객체 존재 신뢰도값이 없음 85 -> 84

    # 이전 단계 객체 인식 결과 초기화
    boxes.clear()  
    confidences.clear()
    class_ids.clear()
    indexes = []

    for detection in outs:
        class_probs = detection[4:]  # 80개 클래스 확률
        class_id = np.argmax(class_probs)  # 가장 높은 확률을 가진 클래스
        conf = class_probs[class_id]  # 신뢰도 계산

        if conf > 0.5:  # 신뢰도 50% 이상만 표시
            # print(detection[0:4]) 출력 결과가 
            # YOLOv3랑 다르게 좌표가 0~1이 아니고 YOLO 이미지 size에 맞춰서 나옴
            center_x = int(detection[0]*x_scale)
            center_y = int(detection[1]*y_scale)
            w = int(detection[2]*x_scale)
            h = int(detection[3]*y_scale)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h]) # 검출한 객체의 바운딩박스 좌표 append
            confidences.append(float(conf)) # 각 객체에 대응하는 확률값 append
            class_ids.append(class_id) # 객체의 인덱스 번호 append

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)     

    # 바운딩 박스 그리기
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f'{str(class_names[class_ids[i]])} : {confidences[i]:.2f}'
            # 클래스 이름 표시
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 화면 출력
    cv2.imshow("YOLOv8 ONNX Detection", frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):  # 'q' 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
