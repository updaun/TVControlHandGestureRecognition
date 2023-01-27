import cv2
import time

webcam = cv2.VideoCapture(0)

width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

prev_time = 0
inference_text = "press 'i' to start inference."
inference_status = False
f = 1

while webcam.isOpened():
    status, frame = webcam.read()

    cur_time = time.time()
    time_diff = cur_time - prev_time
    fps = 1 / time_diff
    prev_time = cur_time

    text = "FPS : %0.1f" % fps

    cv2.putText(frame, text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    cv2.putText(frame, inference_text, (0, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

    if status:
        cv2.imshow("test", frame)

    if inference_status and f <= 30:
        inference_text = f'Do a hand gesture. {f} / 30'
        frames.append(frame)
        f += 1
    elif inference_status and f > 30:
        inference_status = False
        
        
        frames = []
        f = 1

    key = cv2.waitKey(1) & 0xFF

    if key == ord('i'):
        inference_status = True
        frames = []
    elif key == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()