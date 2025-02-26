import object
import cv2

obj = object.Object()
    
# 비디오 파일 로드
cap = cv2.VideoCapture('videoplayback.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = obj.preprocess(frame)

    obj.detection(img,frame)
  
    # 화면 출력
    cv2.imshow("YOLOv8 ONNX Detection", frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):  # 'q' 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
