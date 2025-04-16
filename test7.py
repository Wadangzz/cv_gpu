import cv2
import pymcprotocol as mc
from ultralytics import YOLO

# bounding box의 center가 roi 안에 있는지 판별별
def is_center_in_roi(box, roi):
    return roi[0] <= box[0] <= roi[2] and roi[1] <= box[1] <= roi[3]

if __name__ == "__main__":

    pymc3e = mc.Type3E()
    pymc3e.connect("192.168.24.2", 8000)

    roi = [170, 0 , 430, 480]
    found_in_roi = False

    model = YOLO('./runs/detect/project2_1/weights/best.pt')
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.GaussianBlur(frame, (5, 5), 0)
        sharpening = cv2.addWeighted(frame, 1.5, img, -0.5, 0)
        results = model.predict(sharpening, conf= 0.6, iou=0.5)
        boxes = results[0].boxes
        
        cv2.rectangle(sharpening, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 255), 2)

        if boxes.cls.numel() > 0: # 객체가 감지되면
            for box in boxes.xywh.cpu().numpy()[:2]: # 객체의 중심 좌표를 list compressison
                if is_center_in_roi(box,roi): # 객체의 boungind box가 roi에 있다면
                    pymc3e.batchwrite_wordunits(headdevice="D2001", values=[1])

        cv2.imshow("Inspection", results[0].plot())
        
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(delay) & 0xFF == ord('s'):
            cv2.imwrite('annotated_image.jpg', results[0].plot())

    cap.release()
    cv2.destroyAllWindows()
