import MyModule.myhandlandmark as ml
import MyModule.object as object
import cv2
import sqlite3
import time
from ultralytics import YOLO
from ultralytics.engine.results import Results


db = 'C:/Users/wadangzz/Desktop/Wadangzz/wadangzz/PlcModbus/plc_data.db'
model = YOLO('./MyModule/yolov8l.pt')
# hand = ml.handtracking()
# obj = object.Object()

while True:

    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT address, value FROM DigitalTags WHERE address = ?", ('M100',))
        row = cursor.fetchone()
        print(row)
        
        if row[1] == 1:
            cap = cv2.VideoCapture('./test/highway.mp4')
            fps = cap.get(cv2.CAP_PROP_FPS)
            delay = int(1000 / fps)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                height, width, _ = frame.shape
                img = cv2.GaussianBlur(frame, (5, 5), 0)
                sharpening = cv2.addWeighted(frame, 1.5, img, -0.5, 0)
                results = model.predict(img, conf = 0.3, iou = 0.5)

                cv2.imshow("YOLOv8 Detection", results[0].plot())
                if cv2.waitKey(delay) & 0xFF == ord('q'):  # 'q' 누르면 종료
                    break

            cap.release()
            cv2.destroyAllWindows()
        else:
            time.sleep(1)
            continue