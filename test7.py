import cv2
import pymcprotocol as mc
from ultralytics import YOLO

pymc3e = mc.Type3E()
pymc3e.connect("192.168.24.2", 8000)

model = YOLO('./runs/detect/project2/weights/best.pt')
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.GaussianBlur(frame, (5, 5), 0)
    sharpening = cv2.addWeighted(frame, 1.5, img, -0.5, 0)
    results = model.predict(sharpening, conf=0.6, iou=0.5)
    boxes = results[0].boxes

    if boxes.cls.numel() > 0:
        
        pymc3e.batchwrite_wordunits(headdevice="D2001", values=[1]) 

    cv2.imshow("YOLOv8 Detection", results[0].plot())
    

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(delay) & 0xFF == ord('s'):
        cv2.imwrite('annotated_image.jpg', results[0].plot())

cap.release()
cv2.destroyAllWindows()
