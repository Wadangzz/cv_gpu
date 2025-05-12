"""
📌 비상 정지 시스템 (손 감지 기반)
- MediaPipe를 이용해 실시간으로 손을 감지하고
- ROI 안에 손이 들어오면 Mitsubishi PLC(D8)에 비상정지 신호를 전달
- FastAPI로 웹에서 실시간 영상 확인 가능
"""

import cv2
import time
import threading
import uvicorn
import pymcprotocol as mc
import MyModule.myhandlandmark as ml
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

# 최신 프레임 저장 (웹 스트리밍용)
latest_frame = None

# 프레임을 JPEG로 인코딩하여 스트리밍 반환
def encoding():
    while True:
        if latest_frame is not None:
            _, buffer = cv2.imencode('.jpg', latest_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)

# FastAPI 라우트: MJPEG 스트리밍
@app.get("/video")
def video_feed():
    return StreamingResponse(encoding(), media_type='multipart/x-mixed-replace; boundary=frame')

# FastAPI 서버 실행 (서브스레드에서 실행)
def start_fastapi(app):
    uvicorn.run(app, host="0.0.0.0", port=9001)

if __name__ == "__main__":

        # FastAPI 실행을 위한 서브스레드 시작
    inspection_thread = threading.Thread(target=start_fastapi, args=(app,), daemon=True)
    inspection_thread.start()

    # PLC 통신 초기화 (Type3E 방식 사용, Q03UDE와의 연결)
    pymc3e = mc.Type3E()
    pymc3e.connect("192.168.24.2", 8001)  # PLC IP와 포트

    # 손 추적기 초기화
    hand = ml.handtracking()

    # 카메라 열기 (1번 장치)
    cap = cv2.VideoCapture(1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임 전처리 및 손 인식
        img = hand.preprogress(frame)
        detection_result = hand.detector.detect(img)
        annotated_img, width, height = hand.draw_landmarks_on_image(img.numpy_view(), detection_result)
        hand_landmarks_list = detection_result.hand_landmarks

        # ROI 표시
        cv2.polylines(annotated_img, [hand.roi_polygon], isClosed=True, color=(255, 0, 0), thickness=2)
        
        # 손이 감지되었을 때
        if len(hand_landmarks_list) > 0: # 손이 감지 됬을 때
            hand_points = []
            for i in range(21):
                hand_points.append(i) # Landmark index
            found_inside = False

            for point_idx in hand_points: # 
                pt = hand_landmarks_list[0][point_idx] # 감지된 손의 Landmark 좌표
                pt_x = int(pt.x * 640)
                pt_y = int(pt.y * 480)

                if hand.is_point_in_roi(pt_x, pt_y): # Landmark 좌표가 ROI 안에 있는지 확인
                    found_inside = True
                    break

            if found_inside: # Landmark 좌표가 ROI 안에 있을 때
                pymc3e.batchwrite_wordunits(headdevice="D8", values=[1]) # PLC D8에 1을 쓴다
                cv2.putText(annotated_img, "Warning : Hands detected. Emergence stop", 
                            (45, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # 최신 프레임 저장 (웹 스트리밍용)
        latest_frame = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)

        # 화면 출력
        cv2.imshow('CCTV', latest_frame)
       
        if cv2.waitKey(delay) & 0xFF == ord('q'):  # 'q' 누르면 종료
            break
        elif cv2.waitKey(delay) & 0xFF == ord('s'):
            cv2.imwrite('annotated_img.jpg', annotated_img)

    cap.release()
    cv2.destroyAllWindows()
