import numpy as np
import cv2
from ultralytics import YOLO

model = YOLO('./runs/detect/project2_1/weights/best.pt')
# hand = ml.handtracking()
# obj = object.Object()
    
frame = cv2.imread('C:/Users/user/Documents/GitHub/cv_gpu/dataset/Project_test2/images/train/000025.jpg')

height, width, _ = frame.shape
img = cv2.GaussianBlur(frame, (5, 5), 0)

sharpening = cv2.addWeighted(frame, 1.5, img, -0.5, 0)
results = model.predict(sharpening, conf = 0.3, iou = 0.5)
bbox = results[0].boxes.xyxy.tolist()

    # 화면 출력
print(bbox)

x1, y1, x2, y2 = map(int, bbox[0])
roi = frame[y1:y2,x1:x2]
roi_display = roi.copy()

edges = cv2.Canny(roi, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 4. 가장 큰 외곽선 → 회전된 사각형
cnt = max(contours, key=cv2.contourArea)
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.intp(box)
(center), (w, h), angle = rect
if w < h:
    angle = angle
  # <- 여기서 기울기 얻음
print(angle)

cv2.imshow("YOLOv8 Detection", results[0].plot())
# cv2.imshow('Hand_Tracking',cv2.cvtColor(annotated_img,cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

# 외곽선 그리기 (cnt는 contour 하나이므로 [cnt] 리스트로 감쌈)
cv2.drawContours(roi_display, [cnt], -1, (0, 255, 0), 2)  # 초록색, 두께 2

cv2.drawContours(roi_display, [box], 0, (0, 0, 255), 2)

# 결과 보여주기
cv2.imshow("ROI with contour", roi_display)
cv2.waitKey(0)
cv2.destroyAllWindows()

rotation_matrix = cv2.getRotationMatrix2D(center, angle-90, scale=1.0)

# 4. 회전된 이미지 생성 (크기: 원본 ROI 크기)
rotated = cv2.warpAffine(roi, rotation_matrix, (roi.shape[1], roi.shape[0]))

# 5. 결과 보기
cv2.imshow("Rotated ROI", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()


