import cv2
import time

webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

prev_time = 0

while webcam.isOpened():
    status, frame = webcam.read()

    cur_time = time.time()
    time_diff = cur_time - prev_time
    fps = 1 / time_diff
    prev_time = cur_time

    text = "FPS : %0.1f" % fps

    cv2.putText(frame, text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

    if status:
        cv2.imshow("test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()