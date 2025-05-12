# 🧠 cv_gpu

## 📘 프로젝트 개요

**cv_gpu**는 실시간 영상 기반의 품질 검사 및 안전 시스템으로,  
YOLO 객체 인식과 MediaPipe 손 감지를 통해 자동화 라인에서 불량 검출 및 안전 제어를 수행하며,  
PLC 연동과 DB 기록, 웹 시각화까지 포함한 **풀스택 Computer Vision + 제어 시스템**입니다.

anaconda cv_env1.3.yml 가상환경 설치바랍니다.(GPU 가속 CUDA 필수)
  
```
conda env create -f cv_env1.3.yml
# 미쓰비시 mc 프로토콜 Library, MySQL Library 설치
pip install pymcprotocol
pip install pymysql
```
---

## 🔧 주요 기능

### 🎯 1. YOLO 기반 불량 감지 시스템 (`test1_fastapi.py`)
- **YOLOv5** 모델로 제품의 결함(Dust, Scratch)을 실시간 감지.
- **ROI(관심영역)** 안에 객체가 존재할 경우 **PLC D2001 주소에 Write** → 불량 이벤트 발생.
- 감지 결과에 따라 DB(`productnum`)에 `inspection` 상태를 `OK` 또는 `NG`로 기록.
- 검사 완료 시 `OK`, `NG` 테이블로 데이터 분리(이동 + 삭제).

### ✋ 2. MediaPipe 기반 손 감지 비상정지 시스템 (`test2_fastapi.py`)
- **MediaPipe**로 손이 ROI 안에 감지되면 **PLC D8 주소에 1을 Write** → **비상정지 이벤트 호출**.
- 실시간 영상 스트리밍과 감지 상태 표시(텍스트 오버레이).

### 🌐 3. FastAPI 웹 서버
- `/video` 엔드포인트를 통해 **HTTP MJPEG 스트리밍** 제공.
- 불량 감지 화면(`9000 포트`)과 손 감지 화면(`9001 포트`)으로 분리 운영 가능.

---

## 🗂️ 프로젝트 구조

| 파일/폴더 | 설명 |
|-----------|------|
| `test1_fastapi.py` | YOLO 모델 기반 불량 감지, PLC 제어, MySQL 기록 기능 포함 |
| `test2_fastapi.py` | 손 감지 기반 비상정지 트리거 제어 |
| `MyModule/myhandlandmark.py` | MediaPipe 손 감지 클래스 모듈 |
| `runs/` | YOLO 학습 결과 및 weight(`best.pt`) 저장 폴더 |
| `product_db` | 제품 및 검사 결과 기록용 MySQL DB |

---

## 🔄 시스템 동작 흐름

```plaintext
[카메라 영상] → [YOLO 불량 감지]
                  ↳ 불량 감지 시 PLC에 D2001=1 쓰기
                  ↳ DB에 'NG' 기록 및 NG 테이블로 이동
                  ↳ 양품이면 'OK' 테이블로 이동

[MediaPipe 손 감지] → ROI 안에 손이 들어오면
                      ↳ PLC D8에 1 쓰기 (비상정지 신호)

[FastAPI] → HTTP로 실시간 영상 스트리밍 제공
```

---

## 💻 기술 스택

- **Computer Vision**: YOLOv5, OpenCV, MediaPipe
- **제어/통신**: pymcprotocol (PLC 제어), pymysql
- **백엔드 서버**: FastAPI + Uvicorn
- **데이터베이스**: MySQL (`productnum`, `OK`, `NG` 테이블)
