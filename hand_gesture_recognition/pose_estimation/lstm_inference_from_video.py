import cv2
import sys, os
import mediapipe as mp
import numpy as np
import time
import pandas as pd
from modules.utils import createDirectory
from modules.utils import One_Hand_Coordinate_Normalization, One_Hand_Vector_Normalization
from tensorflow.keras.models import load_model

model = load_model('models/hand_gesture_classifier.h5')


# 시퀀스의 길이(30 -> 10)
seq_length = 30

actions = ['volume_up', 'volume_down', '10s_back', '10s_forword', 'stop']


# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3)

# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# fourcc = cv2.VideoWriter_fourcc(*'MPEG')
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

videoFolderPath = os.path.join("dataset", "test")
videoList = os.listdir(videoFolderPath)

targetList = []

created_time = int(time.time())

for videoPath in videoList:
    actionVideoPath = os.path.join(videoFolderPath, videoPath)
    targetList.append(actionVideoPath)

# print(targetList)
total_frame_count = 0
catch_hand_count = 0

predicts = []

for video_info in zip(targetList, videoList):
    target, file_name = video_info
    video_frame_count = 0

    data = []

    print("Now Streaming :", target)
    cap = cv2.VideoCapture(target)

    # 열렸는지 확인
    if not cap.isOpened():
        print("Camera open failed!")
        sys.exit()

    # 웹캠의 속성 값을 받아오기
    # 정수 형태로 변환하기 위해 round
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) # 카메라에 따라 값이 정상적, 비정상적
    if fps != 0:
        delay = round(1000/fps)
    else:
        delay = round(1000/30)

    video_output_dir = os.path.join("dataset","output_test_video")
    createDirectory(video_output_dir)
    saved_video_path = os.path.join(video_output_dir, file_name)

    # 프레임을 받아와서 저장하기
    out = cv2.VideoWriter(saved_video_path, fourcc, fps, (w, h))
    while True:
        ret, img = cap.read()
        
        if not ret:
            break

        total_frame_count += 1
        video_frame_count += 1

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 손을 인식한 경우
        if result.multi_hand_landmarks is not None:
            catch_hand_count += 1
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 2)) # 넘파이 배열 크기 변경
                x_right_label = []
                y_right_label = []
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y] # z축 제거, visibility 제거
            
                
                full_scale = One_Hand_Coordinate_Normalization(joint)
                v, angle_label = One_Hand_Vector_Normalization(joint)

                d = np.concatenate([full_scale, angle_label.flatten()])
                # print(d.shape)
                data.append(d)
                # data = np.append(data, d)
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
                
        else:
            # 손을 인식하지 못한 경우
            pass
            # 그때 그때 0으로 채울것이냐

        
        out.write(img)

        cv2.imshow("img", img)

        if cv2.waitKey(delay) == 27: # esc를 누르면 강제 종료
            break

    cap.release()
    out.release()
    

    while len(data) < seq_length:
        data.append([0]*57)
        # 맨 마지막에 0으로 프레임 맞춰줄 것이냐
    

    data = np.array(data)

    input_data = np.expand_dims(data, axis=0)

    y_pred = model.predict(input_data).squeeze()

    i_pred = int(np.argmax(y_pred))
    conf = y_pred[i_pred]

    action = actions[i_pred]
    print(f"label:{i_pred} predict : {action} conf : {conf}")
    predicts.append(i_pred)
    
    print("\n---------- Finish Video Streaming ----------")



print("\n---------- Finish Save Dataset ----------")

submission = pd.read_csv("dataset/sample_submission.csv")
submission["label"] = predicts
submission.to_csv("dataset/result_submission.csv")

print(f"총 프레임 수는 {total_frame_count} 입니다.")
print(f"총 프레임 중 손을 인식한 프레임의 수는 {catch_hand_count} 입니다.")
print(f"총 프레임 중 손을 인식하지 못한 프레임의 수는 {total_frame_count-catch_hand_count} 입니다.")

cv2.destroyAllWindows()

