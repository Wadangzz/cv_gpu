import cv2

class Detection():

    def __init__(self,model,cap,roi):
        self.model = model
        self.cap = cap

        self.roi = roi
        self.ng = {
            0: {"id": 'Dust',
                "detected" : False},
            1: {"id": 'Scratch',
                "detected" : False}
        }

    def is_center_in_roi(self,box):
        return self.roi[0] <= box[0] <= self.roi[2] and self.roi[1] <= box[1] <= self.roi[3]
    
    def detect(self, frame):
        img = cv2.GaussianBlur(frame, (5, 5), 0)
        sharpening = cv2.addWeighted(frame, 1.5, img, -0.5, 0)
        results = self.model.predict(sharpening, conf= 0.65, iou=0.5)
        boxes = results[0].boxes
        return results[0].plot(), boxes

    def isdetected(self,boxes):   
        if boxes.cls.numel() > 0: # 객체가 1개라도 감지되면
            for box in boxes.xywh.cpu().numpy(): # 객체의 중심 좌표를 list compressison
                if self.is_center_in_roi(box):
                    return True
                else:
                    continue
        return False
     
    def classification(self,boxes):
        col_name = None
        for cls in boxes.cls.cpu().numpy(): # cls ID를 numpy 배열로 변환
            if cls == 0 and self.ng[0]["detected"] == False:
                col_name = self.ng[0]["id"] # Dust
                self.ng[0]["detected"] = True
            elif cls == 1 and self.ng[1]["detected"] == False:
                col_name = self.ng[1]["id"] # Scratch
                self.ng[1]["detected"] = True
        return col_name
  
    def inspection(self,inspection_result):
        if inspection_result == 0: # 양품
            ins_result = 'OK'
        elif inspection_result == 1: # 불량
            ins_result = 'NG'
        return ins_result