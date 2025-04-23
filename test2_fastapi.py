import cv2
import time
import threading
import uvicorn
import pymcprotocol as mc
import MyModule.myhandlandmark as ml
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()
# 최신 프레임 저장용 전역 변수
latest_frame = None

def encoding():
    while True:
        if latest_frame is not None:
            _, buffer = cv2.imencode('.jpg', latest_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)

@app.get("/video")
def video_feed():
    return StreamingResponse(encoding(), media_type='multipart/x-mixed-replace; boundary=frame')

# FastAPI 실행
def start_fastapi(app):
    uvicorn.run(app, host="0.0.0.0", port=9001)

if __name__ == "__main__":

    inspection_thread = threading.Thread(target=start_fastapi,args = (app,),daemon=True)
    inspection_thread.start()

    pymc3e = mc.Type3E()
    pymc3e.connect("192.168.24.2", 8001)
    hand = ml.handtracking()
        
    # 비디오 파일 로드
    cap = cv2.VideoCapture(1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        emerg_stop = pymc3e.batchread_bitunits(headdevice="M504", readsize=1)[0]
        img = hand.preprogress(frame)
        detection_result = hand.detector.detect(img)
        annotated_img, width, height = hand.draw_landmarks_on_image(img.numpy_view(), detection_result)
        hand_landmarks_list = detection_result.hand_landmarks

        cv2.polylines(annotated_img, [hand.roi_polygon], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.imshow('CCTV',cv2.cvtColor(annotated_img,cv2.COLOR_RGB2BGR))
        latest_frame = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
        if len(hand_landmarks_list) > 0:
            hand_points = [4, 7, 11, 15, 20]
            found_inside = False

            for point_idx in hand_points:
                pt = hand_landmarks_list[0][point_idx]
                pt_x = int(pt.x * 640)
                pt_y = int(pt.y * 480)

                if hand.is_point_in_roi(pt_x, pt_y):
                    found_inside = True
                    break

            if (found_inside and emerg_stop == 1):
                print("손이 ROI 안에 있음")
                pymc3e.batchwrite_wordunits(headdevice="D8", values=[1]) 

        if cv2.waitKey(delay) & 0xFF == ord('q'):  # 'q' 누르면 종료
            break
        elif cv2.waitKey(delay) & 0xFF == ord('s'):
            cv2.imwrite('annotated_img.jpg', annotated_img)

    cap.release()
    cv2.destroyAllWindows()
