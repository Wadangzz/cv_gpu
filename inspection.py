import cv2
import time
import threading
import uvicorn
import pymcprotocol as mc
import pymysql
from ultralytics import YOLO
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from MyModule.detection import Detection

app = FastAPI()
# 최신 프레임 저장용 전역 변수
latest_frame = None

# YOLO 검출 결과 전송 encoding
def encoding():
    while True:
        if latest_frame is not None:
            _, buffer = cv2.imencode('.jpg', latest_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.4)

# FastAPI 라우트: MJPEG 스트리밍
@app.get("/video")
def video_feed():
    return StreamingResponse(encoding(), media_type='multipart/x-mixed-replace; boundary=frame')

# FastAPI 실행
def start_fastapi(app):
    uvicorn.run(app, host="0.0.0.0", port=9000)

# 불량품 판별 후 DB 업데이트
def update_ng(cursor,col_name):
    cursor.execute(f"""
                    UPDATE productnum SET {col_name} = %s, inspection = %s
                    WHERE {col_name} = 0 ORDER BY id ASC LIMIT 1""", (1,'NG'))
# 양품 판별 후 DB 업데이트    
def update_ok(cursor):
    cursor.execute("""
                    UPDATE productnum SET inspection = %s 
                    WHERE inspection = 'Not Yet' ORDER BY id ASC LIMIT 1""", ('OK',))

# 제품 검사 결과에 따라 해당 제품코드의 정보를 OK, NG 테이블로 이동    
def movetable(cursor,ins_result):
    cursor.execute(f"""
                    INSERT INTO {ins_result} 
                    (ProductCode, Model, Dust, Scratch, inspection)
                    SELECT ProductCode, Model, Dust, Scratch, inspection
                    FROM productnum WHERE inspection = %s LIMIT 1""", (ins_result,))
    cursor.execute("DELETE FROM productnum WHERE inspection = %s LIMIT 1", (ins_result,))

if __name__ == "__main__":
    
    inspection_thread = threading.Thread(target=start_fastapi,args = (app,),daemon=True)
    inspection_thread.start() # FastAPI 실행을 위한 서브스레드 시작
    
    plc = mc.Type3E() # PLC 통신 초기화 (Type3E 방식 사용, Q03UDE와의 연결)
    plc.connect("192.168.24.2", 8000)

    inspected = False # 검사 완료 여부
    roi = [170, 0 , 430, 480] # ROI 좌표

    INSPECTION_MODE = 36
    INSPECTION_COMPLETE = 37

    with pymysql.connect(
        host = '127.0.0.1',
        user = 'product',
        password = '1122334455',
        database = 'product_db') as conn:
        
        # YOLOv8 모델 로드
        model = YOLO('./runs/detect/project2_1/weights/best.pt')
        cap = cv2.VideoCapture(0) # 카메라 열기 (0번 장치)
        fps = cap.get(cv2.CAP_PROP_FPS) # 초당 프레임 수
        delay = int(1000 / fps) # 프레임 지연 시간

        yolo = Detection(model,roi) # YOLO 객체 생성

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_image, boxes = yolo.detect(frame)

            cv2.rectangle(annotated_image, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 255), 2)

            status = plc.batchread_wordunits(headdevice="D16",readsize=1)[0]
            if status == INSPECTION_MODE:
                with conn.cursor() as cursor:
                    try:
                        if yolo.is_detected_in_roi(boxes):
                            plc.batchwrite_wordunits(headdevice="D2001", values=[1]) # PLC D2001에 1을 쓴다
                            col_name = yolo.classification(boxes)
                            if not col_name == None:
                                update_ng(cursor, col_name) # col_name에 해당하는 불량품 DB 업데이트
                        else:   
                            update_ok(cursor) # 양품 DB 업데이트
                        conn.commit()
                    except Exception as e:
                        print(f"Mysql Error : {e}")
                        conn.rollback()

            if status == INSPECTION_COMPLETE and inspected == False:
                ins_result = None
                for i in range(2):
                    yolo.ng[i]["detected"] = False # 객체 감지 결과를 리셋
                inspected = True
                decision = plc.batchread_wordunits(headdevice="D2001",readsize=1)[0]
                ins_result = yolo.get_inspection_result(decision)
                if not ins_result == None:
                    with conn.cursor() as cursor: # 검사 결과에 따라 해당 제품코드의 정보를 OK, NG 테이블로 이동
                        try:
                            movetable(cursor, ins_result)
                            conn.commit()
                        except Exception as e:
                            print(f"Mysql Error : {e}")
                            conn.rollback()

            if status != INSPECTION_COMPLETE and inspected: # 검사 완료되어 후공정 이동
                inspected = False

            latest_frame = annotated_image
            cv2.imshow("Inspection", annotated_image)

            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(delay) & 0xFF == ord('s'):
                cv2.imwrite('annotated_image.jpg', annotated_image)

        cap.release()
        cv2.destroyAllWindows()