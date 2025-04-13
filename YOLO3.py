import cv2
import numpy as np

class Vision():


    def __init__(self):

        # YOLO 모델 및 설정 파일 경로 (실제 경로로 변경 필요)
        model_cfg = "./darknet-master/cfg/yolov3.cfg"
        model_weights = "./darknet-master/yolov3.weights"
        
        self.outs = [] 
        self.boxes = [] # 객체 바운딩 박스 list
        self.confidences = [] # 객체 확률값 list
        self.class_ids = [] # 객체 인덱스 list
        self.indexes = [] # 겹치는 바운딩 박스 제거 후 남은 box 인덱스 list
        
        self.net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

        # 클래스 이름 파일 (coco names)
        # # List Comprehension
        with open("./darknet-master/data/coco.names", "r") as f: # 파일 열기 r = 읽기 모드
            self.classes = [line.strip() for line in f.readlines()]
            # line.strip() 각줄의 앞뒤 공백 제거
            # YOLO가 탐지한 객체의 번호(class_id)를 실제 이름과 매칭하기 위해 사용.

        # 모델의 모든 레이어 이름을 리스트로 반환 Layer = 합성곱 층
        # conv_(합성곱 Layer),bn_(배치 정규화 Layer),leaky(ReLU 함수),yolo_(YOLO 레이어로 구성)
        layer_names = self.net.getLayerNames()# 네트워크의 마지막 출력 레이어들의 인덱스를 반환( YOLO 출력 Layer 3개 )
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = [(0, 0, 255)] # 바운딩 박스 색깔 (빨간색)
        self.font = cv2.FONT_HERSHEY_SIMPLEX # 폰트 설정

    def Blob(self,_frame):

        # YOLO 입력으로 사용할 이미지 전처리 (크기 조정, 정규화 등)
        # 4차원 배열 반환
        # (배치 크기, RGB 채널 수 = 3, 416,416)
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob) # 변환한 이미지 입력


    def DetectYOLO(self,_height,_width):


        # 이전 프레임 객체 검출 정보 초기화
        self.boxes.clear()  
        self.confidences.clear()
        self.class_ids.clear()
        self.indexes = []  # NMS 결과도 초기화


        # 검출 결과 처리
        for out in self.outs: # 객체 탐지 결과 전체 반복
            for detection in out:
                scores = detection[5:] # outs의 5~85 확률값 채운다
                class_id = np.argmax(scores) # 확률값이 가장 큰 놈의 index
                confidence = scores[class_id]
                if confidence > 0.5: # 신뢰도 0.5 이상만 검출
                    center_x = int(detection[0] * _width)
                    center_y = int(detection[1] * _height)
                    w = int(detection[2] * _width)
                    h = int(detection[3] * _height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    self.boxes.append([x, y, w, h]) # 검출한 객체의 바운딩박스(??) append
                    self.confidences.append(float(confidence)) # 각 객체 바운딩박스에 대응하는 확률값 append
                    self.class_ids.append(class_id) # 객체의 인덱스 번호 append

        # Non-Maximum Suppression (겹치는 바운딩 박스 제거)
        self.indexes = cv2.dnn.NMSBoxes(self.boxes, self.confidences, 0.5, 0.4)
           
    def Bbox(self,_frame):
        
        for i in range(len(self.boxes)):
            if i in self.indexes:
                x, y, w, h = self.boxes[i]
                label = str(self.classes[self.class_ids[i]])
                color = self.colors[0] # 빨간색
                cv2.rectangle(_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(_frame, label, (x, y - 10), self.font, 0.5, color, 1)


    def ForwardPropagation(self):

        self.outs = self.net.forward(self.output_layers)
        # 네트워크 순방향 전파
        # 객체 탐지 결과 리스트 반환
        # 출력리스트 각각의 인덱스는 (객체 갯수, 85) 배열 85 = YOLO 출력 벡터
        # 0 = x좌표 중심
        # 1 = y좌표 중심
        # 2 = x 너비
        # 3 = y 높이
        # 4 = 객체 존재 신뢰도 
        # 0~4 모두 범위는 0~1
        # 5~85 각 클래스의 대한 확률

vv = Vision()

# 비디오 캡쳐 또는 이미지 읽기
cap = cv2.VideoCapture("./wadangzz/highway.mp4") # 또는 cv2.imread("image.jpg")
fps = cap.get(cv2.CAP_PROP_FPS) # 프레임 수 구하기
delay = int(1000/fps)

while cap.isOpened(): 

    ret, frame = cap.read() # frame = 이미지
    if not ret: 
        break

    height, width, channels = frame.shape # frame 배열의 크기

    # 이미지 전처리 (크기 조정, 정규화 등)
    vv.Blob(frame)

    # 네트워크 순방향 전파
    vv.ForwardPropagation()
   
    # 검출 결과 처리
    vv.DetectYOLO(height,width)

    # 검출된 객체 화면에 표시
    vv.Bbox(frame)

    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(delay) & 0xff == 27: # ESC 키 종료
        break

cap.release()
cv2.destroyAllWindows()