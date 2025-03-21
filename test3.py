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

            inspections = []
            annotated_frame = []

            cap = cv2.VideoCapture(0)
            fps = cap.get(cv2.CAP_PROP_FPS)
            delay = int(1000 / fps)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                height, width, _ = frame.shape
                for i in range(5):
                    inspections.append(frame)

                for imgs in inspections:
                    img = cv2.GaussianBlur(imgs, (5, 5), 0)
                    sharpening = cv2.addWeighted(imgs, 1.5, img, -0.5, 0)
                
                    results = model.predict(sharpening, conf=0.3, iou=0.5)
                    annotated_frame.append(results[0].plot())
                    time.sleep(1)

                print(annotated_frame)
                
                # for result in annotated_frame:
                #     print(result)
                #     cv2.imshow("YOLOv8 Detection", result)
                #     time.sleep(1)
                       
                break
          
            # cv2.destroyAllWindows()

                    # cv2.imshow("YOLOv8 Detection", annotated_frame)
                    # if cv2.waitKey(delay) & 0xFF == ord('q'):
                    #     break

            # cap.release()
            # cv2.destroyAllWindows()
            
            cap.release()
        else:
            print('비전 검사 대기중')
            time.sleep(1)
            continue