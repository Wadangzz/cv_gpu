import cv2
import time
import threading
import uvicorn
import pymcprotocol as mc
import pymysql
from ultralytics import YOLO
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()
# 최신 프레임 저장용 전역 변수
latest_frame = None

# bounding box의 center가 roi 안에 있는지 판별(true, false 반환)
def is_center_in_roi(box, roi):
    return roi[0] <= box[0] <= roi[2] and roi[1] <= box[1] <= roi[3]

# YOLO 검출 결과 전송 encoding
def encoding():
    while True:
        if latest_frame is not None:
            _, buffer = cv2.imencode('.jpg', latest_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.4)

@app.get("/video")
def video_feed():
    return StreamingResponse(encoding(), media_type='multipart/x-mixed-replace; boundary=frame')

# FastAPI 실행
def start_fastapi(app):
    uvicorn.run(app, host="0.0.0.0", port=9000)

if __name__ == "__main__":
    
    inspection_thread = threading.Thread(target=start_fastapi,args = (app,),daemon=True)
    inspection_thread.start()
    
    pymc3e = mc.Type3E()
    pymc3e.connect("192.168.24.2", 8000)

    roi = [170, 0 , 430, 480]
    found_in_roi = False

    with pymysql.connect(
        host = '127.0.0.1',
        user = 'product',
        password = '*********',
        database = 'product_db') as conn:

        ng = {
            0: {"id": 'Dust',
                "detected" : False},
            1: {"id": 'Scratch',
                "detected" : False}
            }

        # inspect_count = 0
        inspected = False

        INSPECTION_MODE = 36
        INSPECTION_COMPLETE = 37
        # MAX_INSPECT_COUNT = 10

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
            results = model.predict(sharpening, conf= 0.65, iou=0.5)
            boxes = results[0].boxes
            
            cv2.rectangle(sharpening, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 255), 2)
            # D16이 36일 때 검사 진행 중
            status = pymc3e.batchread_wordunits(headdevice="D16",readsize=1)[0] 

            if status == INSPECTION_MODE:# and inspect_count < MAX_INSPECT_COUNT:
                with conn.cursor() as cursor:
                    try:
                        col_name = None
                        if boxes.cls.numel() > 0: # 객체가 1개라도 감지되면
                            for box in boxes.xywh.cpu().numpy()[:2]: # 객체의 중심 좌표를 list compressison
                                if is_center_in_roi(box,roi): # 객체의 boungind box가 roi 안에 있다면
                                    pymc3e.batchwrite_wordunits(headdevice="D2001", values=[1]) # PLC D2001에 1을 쓴다
                            for cls in boxes.cls.cpu().numpy(): # cls ID를 numpy 배열로 변환
                                if cls == 0 and ng[0]["detected"] == False:
                                    col_name = ng[0]["id"] # Dust
                                    ng[0]["detected"] = True
                                elif cls == 1 and ng[1]["detected"] == False:
                                    col_name = ng[1]["id"] # Scratch
                                    ng[1]["detected"] = True
                                if not col_name == None:
                                    cursor.execute(
                                        f"""
                                        UPDATE productnum SET {col_name} = %s
                                        WHERE inspection = 'Not Yet' AND {col_name} = 0 ORDER BY id ASC LIMIT 1""", (1,))
                            cursor.execute(
                                """
                                UPDATE productnum SET inspection = %s 
                                WHERE inspection = 'Not Yet' ORDER BY id ASC LIMIT 1""", ('NG',))
                            conn.commit()  # 여기서 한 번만 commit
                        else:   
                            cursor.execute(
                                """
                                UPDATE productnum SET inspection = %s 
                                WHERE inspection = 'Not Yet' ORDER BY id ASC LIMIT 1""", ('OK',))
                            conn.commit()
                    except Exception as e:
                        print(f"Mysql Error : {e}")
                        conn.rollback()

            if status == INSPECTION_COMPLETE and inspected == False:
                # inspect_count = 0
                ins_result = None
                for i in range(2):
                    ng[i]["detected"] = False # 객체 감지 결과를 리셋
                inspected = True
                inspection_result = pymc3e.batchread_wordunits(headdevice="D2001",readsize=1)[0]
                if inspection_result == 0: # 양품
                    ins_result = 'OK'
                elif inspection_result == 1: # 불량
                    ins_result = 'NG'
                if not ins_result == None:
                    with conn.cursor() as cursor: # 검사 결과에 따라 해당 제품코드의 정보를 OK, NG 테이블로 이동
                        try:
                            cursor.execute(
                                f"""
                                INSERT INTO {ins_result} 
                                (ProductCode, Model, Dust, Scratch, inspection)
                                SELECT ProductCode, Model, Dust, Scratch, inspection
                                FROM productnum WHERE inspection = %s LIMIT 1""", (ins_result,))
                            cursor.execute(    
                                "DELETE FROM productnum WHERE inspection = %s LIMIT 1", (ins_result,))
                            conn.commit()
                        except Exception as e:
                            print(f"Mysql Error : {e}")
                            conn.rollback()                  

            if status != INSPECTION_COMPLETE and inspected: # 검사 완료되어 후공정 이동
                inspected = False

            annotated_image = results[0].plot()
            latest_frame = results[0].plot()
            cv2.imshow("Inspection", annotated_image)

            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(delay) & 0xFF == ord('s'):
                cv2.imwrite('annotated_image.jpg', annotated_image)

        cap.release()
        cv2.destroyAllWindows()
