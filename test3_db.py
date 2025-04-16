import MyModule.myhandlandmark as ml
import MyModule.object as object
import pymcprotocol as mc
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results


pymc3e = mc.Type3E()
pymc3e.connect("192.168.24.2", 8001)
hand = ml.handtracking()
# obj = object.Object()
    

# 비디오 파일 로드
cap = cv2.VideoCapture(1)
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width, _ = frame.shape
    # img = cv2.GaussianBlur(frame, (5, 5), 0)
    # sharpening = cv2.addWeighted(frame, 1.5, img, -0.5, 0)
    # results = model.predict(frame, conf = 0.7, iou = 0.5)

    # boxes = results[0].boxes # tensor 형태의 bounding box 정보 ( box 좌표 4개, 확률, 객체 정보
    # person_boxes = boxes[boxes.cls == 0]q
    # # YOLO model plot에 맞게 Results class로 변환
    # person_result = Results(q
    #     orig_img = results[0].orig_img,  # 원본 이미지 유지
    #     path = results[0].path,  # 이미지 경로 유지
    #     names = results[0].names,  # 클래스 ID -> 클래스명 매핑 유지
    #     boxes = person_boxes.data[:len(person_boxes)]  # "person" 클래스만 포함된 bbox tensor
    # ) 

    img = hand.preprogress(frame)
    detection_result = hand.detector.detect(img)
    annotated_img, width, height = hand.draw_landmarks_on_image(img.numpy_view(), detection_result)
    hand_landmarks_list = detection_result.hand_landmarks

    # img = obj.preprocess(frame)
    # obj.detection(img,frame)

    # cv2.imshow("YOLOv8 Detection", results[0].plot())
    cv2.polylines(annotated_img, [hand.roi_polygon], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.imshow('Hand_Tracking',cv2.cvtColor(annotated_img,cv2.COLOR_RGB2BGR))
    if len(hand_landmarks_list) > 0:
        hand_points = [4, 7, 11, 15, 20]
        found_inside = False

        for point_idx in hand_points:
            pt = hand_landmarks_list[0][point_idx]
            pt_x = int(pt.x * 640)
            pt_y = int(pt.y * 480)

            if hand.is_point_in_roi(pt_x, pt_y):
                found_inside = True
                break

        if found_inside:
            print("손이 ROI 안에 있음")
            pymc3e.batchwrite_wordunits(headdevice="D8", values=[1]) 

    if cv2.waitKey(delay) & 0xFF == ord('q'):  # 'q' 누르면 종료
        break
    elif cv2.waitKey(delay) & 0xFF == ord('s'):
        cv2.imwrite('annotated_img.jpg', annotated_img)

cap.release()
cv2.destroyAllWindows()
