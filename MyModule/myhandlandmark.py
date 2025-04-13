import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


class handtracking:

    def __init__(self):

        self.MARGIN = 10 # pixel수
        self.FONT_SIZE = 1
        self.FONT_THICKNESS = 1
        self.HANDEDNESS_TEXT_COLOR = (88,205,54) # RGB 색깔

        self.num_hands= 9
        self.min_hand_detection_confidence = 0.5
        self.min_hand_presence_confidence = 0.5
        self.min_tracking_confidence = 0.5
        self.roi_polygon = np.array([
                            [0, 0],
                            [312, 0],
                            [640, 312],
                            [640, 480],
                            [100, 480]
                        ], dtype=np.int32)

        self.options = vision.HandLandmarkerOptions(
            base_options = python.BaseOptions(model_asset_path = "./MyModule/hand_landmarker.task"),
            num_hands = self.num_hands,
            min_hand_detection_confidence = self.min_hand_detection_confidence,
            min_hand_presence_confidence = self.min_hand_presence_confidence,
            min_tracking_confidence = self.min_tracking_confidence,
            )

        self.detector = vision.HandLandmarker.create_from_options(self.options)

    def preprogress(self,frame):  
        
        image = cv2.resize(frame,(640,480))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format = mp.ImageFormat.SRGB, data = image)

        return image
    
    def draw_landmarks_on_image(self,rgb_image, detection_result):

        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        if not hand_landmarks_list:  # 손이 감지되지 않은 경우 예외 처리
            return annotated_image, 0, 0

        for i in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[i]
            handedness = handedness_list[i]
            
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
                
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_utils.DrawingSpec(color=(40, 100, 230), thickness=2, circle_radius=2),  # 랜드마크 스타일
                solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2))  # 연결선 스타일
    
                
            # height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]

            min_x = min(x_coordinates)
            max_x = max(x_coordinates)
            min_y = min(y_coordinates)
            max_y = max(y_coordinates)

            # 너비와 높이 계산
            width = max_x - min_x
            height = max_y - min_y
            # text_x = int(min(x_coordinates) * width)
            # text_y = int(min(y_coordinates) * height) - self.MARGIN

            # cv2.putText(annotated_image,f'{handedness[0].category_name}',
            #             (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
            #             self.FONT_SIZE, self.HANDEDNESS_TEXT_COLOR,self.FONT_THICKNESS, cv2.LINE_AA)
            
        return annotated_image, width, height
    
    def is_point_in_roi(self,x, y):
        return cv2.pointPolygonTest(self.roi_polygon, (x, y), False) >= 0