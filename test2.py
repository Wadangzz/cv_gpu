import MyModule.myhandlandmark as ml
import MyModule.object
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

model = YOLO('./MyModule/mytrained.pt')
# hand = ml.handtracking()
# obj = object.Object()
    
# 비디오 파일 로드
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.predict(frame, conf = 0.5, iou = 0.3)
    # boxes = results[0].boxes # tensor 형태의 bounding box 정보 ( box 좌표 4개, 확률, 객체 정보 )
    # person_boxes = boxes[boxes.cls == 0]

    # # YOLO model plot에 맞게 Results class로 변환
    # person_result = Results(
    #     orig_img = results[0].orig_img,  # 원본 이미지 유지
    #     path = results[0].path,  # 이미지 경로 유지
    #     names = results[0].names,  # 클래스 ID -> 클래스명 매핑 유지
    #     boxes = person_boxes.data[:len(person_boxes)]  # "person" 클래스만 포함된 bbox tensor
    # ) 

    # img = hand.preprogress(frame)

    # detection_result = hand.detector.detect(img)

    # annotated_img = hand.draw_landmarks_on_image(img.numpy_view(), detection_result)

    # img = obj.preprocess(frame)

    # obj.detection(img,frame)
  
    # 화면 출력
    cv2.imshow("YOLOv8 Detection", results[0].plot())
    # cv2.imshow('Hand_Tracking',cv2.cvtColor(annotated_img,cv2.COLOR_RGB2BGR))

    if cv2.waitKey(delay) & 0xFF == ord('q'):  # 'q' 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
