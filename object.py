import onnxruntime as ort
import numpy as np
import cv2

class Object():
    
    def __init__(self):

        # COCO 클래스 읽기
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()] # 리스트 컴프리헨션 문법

        # ONNX YOLO 모델 로드 providers=["CUDAExecutionProvider"] CUDA 환경
        self.session = ort.InferenceSession("best.onnx", providers=["CUDAExecutionProvider"])

        print(self.session.get_providers())

        # 입력 텐서 정보 가져오기
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.x_scale = 0 
        self.y_scale = 0

    def preprocess(self,_frame):
        
    # YOLO 모델 입력 크기로 전처리
        size = 640 # 320 416 640 1280
        img = cv2.resize(_frame, (size, size)) 
        img = img / 255.0  # 정규화 0~1
        img = img.transpose(2, 0, 1)
        # (H, W, C) → (C, H, W) 
        # frame의 attribute 순서를 RGB채널, 높이, 너비 순으로 변환
        img = np.expand_dims(img, axis=0).astype(np.float32)  # (1, C, H, W)
        # 1차원 추가해서 4차원 텐서 형태로 변환
        
        height, width, _ = _frame.shape
        self.x_scale, self.y_scale = width/size, height/size
        return img
    
    def detection(self,_img,_frame):

        boxes = []
        confidences = []
        class_ids = []
        indexes = []

        # 모델 예측 실행
        outputs = self.session.run(self.output_names, {self.input_name: _img})
        outs = outputs[0][0].T  # (84, 3549) → (3549, 84)
        # YOLOv3에 있는 4번째 index 객체 존재 신뢰도값이 없음 85 -> 84

        for detection in outs:
            class_probs = detection[4:]  # 80개 클래스 확률
            class_id = np.argmax(class_probs)  # 가장 높은 확률을 가진 클래스
            conf = class_probs[class_id]  # 신뢰도 계산

            if conf > 0.5:  # 신뢰도 50% 이상만 표시
                # print(detection[0:4]) 출력 결과가 
                # YOLOv3랑 다르게 좌표가 0~1이 아니고 YOLO 이미지 size에 맞춰서 나옴
                center_x = int(detection[0]*self.x_scale)
                center_y = int(detection[1]*self.y_scale)
                w = int(detection[2]*self.x_scale)
                h = int(detection[3]*self.y_scale)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h]) # 검출한 객체의 바운딩박스 좌표 append
                confidences.append(float(conf)) # 각 객체에 대응하는 확률값 append
                class_ids.append(class_id) # 객체의 인덱스 번호 append

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) # 겹치는 박스 제거


        # 바운딩 박스 그리기
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = f'{str(self.classes[class_ids[i]])} : {confidences[i]:.2f}'
                # 클래스 이름 표시
                cv2.rectangle(_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
