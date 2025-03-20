import MyModule.myhandlandmark as ml
import MyModule.object as object
import cv2
import sqlite3
import time
from ultralytics import YOLO
from ultralytics.engine.results import Results


db = 'C:/Users/user/Documents/GitHub/wadangzz/PlcModbus/plc_data.db'
model = YOLO('./MyModule/yolov8l.pt')
# hand = ml.handtracking()
# obj = object.Object()

while True:

    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT address, value FROM DigitalTags WHERE address = ?", ('M16',))
        row = cursor.fetchone()
        print(row)
        
        if row[1] == 1:
            run_detection = True
        else:
            print('비전 검사 대기중')
            time.sleep(1)
            continue

    if run_detection:
        cap = cv2.VideoCapture(0)
        
        # 키 입력 종료 플래그
        should_exit = False
        
        while cap.isOpened() and not should_exit:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # 프레임 처리
            img = cv2.GaussianBlur(frame, (5, 5), 0)
            sharpening = cv2.addWeighted(frame, 1.5, img, -0.5, 0)
            
            results = model.predict(sharpening, conf=0.3, iou=0.5)
            annotated_frame = results[0].plot()
            
            # 결과 표시
            cv2.imshow("YOLOv8 Detection", annotated_frame)
            
            # 'q' 키를 누르면 종료하도록 설정
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                should_exit = True
                print("'q' 키가 눌렸습니다. 프로그램을 종료합니다.")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # 메인 루프도 종료
        if should_exit:
            break