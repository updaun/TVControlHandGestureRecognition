import os
import cv2

video_dir_path = "dataset/train"

frames = []

for i in os.listdir(video_dir_path):
    video_path = os.path.join(video_dir_path, i)

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while cap.isOpened():

        ret, img = cap.read()

        if not ret:
            break
        
        cv2.imshow("img", img)
        if cv2.waitKey(1) == ord('q'):
            break

        frame_count += 1

    frames.append(frame_count)
print(frames)
print(set(frames))