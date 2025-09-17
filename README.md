# 🧠 cv_gpu
![image](https://github.com/user-attachments/assets/478da94d-e97d-4038-90a4-9be0680d7c22)


## 📘 프로젝트 개요
> 
**cv_gpu**는 실시간 영상 기반의 품질 검사 및 안전 시스템으로,  
YOLO 객체 인식과 MediaPipe 손 감지를 통해 자동화 라인에서 불량 검출 및 사고방지 비상정지를 수행하며,  
PLC 연동과 DB 기록, 웹 시각화까지 포함한 **Computer Vision + 제어 시스템**입니다.
> 
anaconda cv_env1.3.yml 가상환경 + 추가 라이브러리 설치바랍니다.(GPU 가속 CUDA 필수)   
[**CUDA 버전에 맞는 pytorch는 개별 설치해야합니다.**](https://pytorch.org/get-started/locally/)
>   
```
conda env create -f cv_env1.3.yml
```
> 
```
# 미쓰비시 mc 프로토콜 Library, MySQL Library 설치
pip install pymcprotocol
pip install pymysql
```
>
가상환경 설치가 잘 됐다면
test_version.py 실행하여 numpy, cv2, torch, onnx 버전 확인 밎
pytorch CUDA 사용 가능 여부 체크 할 수 있습니다.
>
```python
import numpy as np
import cv2
import torch
import onnxruntime as ort

print(np.__version__)
print(cv2.__version__)
print(torch.__version__)
print(ort.__version__)


print("CUDA available (PyTorch):", torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(ort.get_device())



출력

1.26.3
4.11.0
2.6.0+cu118
1.16.3
CUDA available (PyTorch): True
NVIDIA GeForce @@@@@@@(사용중인 GPU 모델명)
GPU
```
---

## ❓ Why YOLO & MediaPipe?
>     
프로젝트의 핵심 요구사항은 "**실시간 불량 감지 및 사고방지 비상정지**"였습니다.  
YOLOv8은 CNN 기반의 객체 감지 모델 중에서도 **가벼우면서도 정확도와 속도 균형이 뛰어나며**   
간단한 Labeling 작업으로 학습용 Dataset을 쉽게 생성할 수 있었기 때문에 채택하였습니다.   
>    
또한 손 감지에는 MediaPipe를 활용했는데, 이는 복잡한 학습 없이도  
**관절 기반 좌표 추정이 가능하고, CPU 환경에서도 안정적인 성능**을 보이는 점에서  
***실시간성과 신뢰성이 중요한 안전 제어 시스템**에 적합하다고 판단했습니다.   
> 
---

## 🔧 주요 기능
> 
### 🎯 1. YOLO 기반 불량 감지 시스템 (`test1_fastapi.py`)
- **YOLOv8** 모델로 제품의 결함(Dust, Scratch)을 실시간 감지
- **ROI(관심영역)** 안에 객체가 존재할 경우 **PLC D2001 주소에 Write** → 불량 이벤트 발생
- 감지 결과에 따라 DB(`productnum`)에 `inspection` 상태를 `OK` 또는 `NG`로 기록
- 검사 완료 시 `OK`, `NG` 테이블로 데이터 분리(이동 + 삭제)

### ✋ 2. MediaPipe 기반 손 감지 비상정지 시스템 (`test2_fastapi.py`)
- **MediaPipe**로 손이 ROI 안에 감지되면 **PLC D8 주소에 1을 Write** → **비상정지 이벤트 호출**
- 실시간 영상 스트리밍과 감지 상태 표시

### 🌐 3. FastAPI 웹 서버
- `/video` 엔드포인트를 통해 **HTTP MJPEG 스트리밍** 제공
- 불량 감지 화면(`9000 포트`)과 손 감지 화면(`9001 포트`)으로 분리 운영
> 
---

## 🗂️ 프로젝트 구조
> 
| 파일/폴더 | 설명 |
|-----------|------|
| `test1_fastapi.py` | YOLO 모델 기반 불량 감지, PLC 제어, MySQL 기록 기능 포함 |
| `test2_fastapi.py` | 손 감지 기반 비상정지 트리거 제어 |
| `MyModule/myhandlandmark.py` | MediaPipe 손 감지 클래스 모듈 |
| `runs/` | YOLO 학습 결과 및 weight(`best.pt`) 저장 폴더 |
| `product_db` | 제품 및 검사 결과 기록용 MySQL DB |
> 
---

## 🔄 시스템 동작 흐름
> 
| **구분**                | **동작 내용** |
|-------------------------|----------------|
| **카메라 영상**         | YOLO로 실시간 불량 감지 수행<br>불량 감지 시 `PLC D2001 = 1` 신호 전송<br>DB에 `'NG'` 기록 후 NG 테이블로 이동<br>양품이면 `'OK'` 테이블로 이동 |
| **MediaPipe 손 감지** | 손이 ROI 안에 들어오면 `PLC D8 = 1` 비상정지 신호 발생 |
| **FastAPI 웹 스트리밍** | `/video` 엔드포인트에서 실시간 MJPEG 영상 스트리밍 제공 |
> 
---

## 🐞 Trouble Shooting

### 📸 1. 밝기 변화에 따른 객체 감지 실패

**문제**  
동일 조도의 이미지로만 학습한 결과,  
**조명이나 밝기 조건이 달라지면 객체 감지가 실패**하는 문제가 발생함

**해결방법**  
다양한 밝기 조건을 반영한 **데이터 증강 이미지셋을 추가 생성**하고,  
이를 활용하여 YOLO 모델을 **추가 학습(Fine-Tuning)** 진행

**결과**  
실시간 환경에서도 **감지 신뢰도 0.65 이상을 안정적으로 유지**할 수 있었습니다.

---

### ✋ 2. 손 감지 오작동 (깊이 정보 부재)

**문제**  
MediaPipe는 2D 손 위치만 감지하기 때문에,  
**실제 장비와의 거리와 무관하게 비상정지가 발생하는 문제**가 있었습니다.

**해결방법**  
PyTorch 기반의 **MiDaS 상대 깊이 추정 모델**을 적용하여,  
손의 거리 정보를 **이미지 기반으로 추정**하려 시도했습니다.

**결과**  
상대 깊이는 추정 가능했지만, **실제 공간 거리와 정확히 매핑되지 않아 적용에 한계**가 있었습니다.

➡️ **향후 계획**
- 양안 카메라 기반 **스테레오 비전 3D 추정 기법 도입**
- 또는 **이미지 캘리브레이션** 적용

---
