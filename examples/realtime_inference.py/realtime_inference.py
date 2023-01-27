import os
import cv2
import numpy as np
import pandas as pd
import time
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

IMG_SIZE = 160
BATCH_SIZE = 1
WEIGHTS_PATH = 'D:/Assignments/공모전/dacon/TVControlHandGestureRecognition/examples/baseline/template/weights/resnet3d18/best_epoch3_1.2364.pth'

class CustomDataset(Dataset):
    def __init__(self, frames, label_list):
        self.frames = frames
        self.label_list = label_list
        
    def __getitem__(self, index):
        frames = self.frames
        
        if self.label_list is not None:
            label = self.label_list[index]
            return frames, label
        else:
            return frames
        
    def __len__(self):
        return 1

def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for videos in tqdm(iter(test_loader)):
            videos = videos.to(device)
            
            logit = model(videos)

            preds += logit.argmax(1).detach().cpu().numpy().tolist()
    return preds

# -- settings
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# -- model
model = models.video.r3d_18(weights='R3D_18_Weights.DEFAULT')
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)

weights_path = WEIGHTS_PATH
model.load_state_dict(torch.load(weights_path, map_location=device))

webcam = cv2.VideoCapture(0)

width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# writer = cv2.VideoWriter('./output.mp4', fourcc, 30, (width, height))

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
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = img / 255.
        frames.append(img)
        f += 1
    elif inference_status and f > 30:
        inference_status = False
        
        frames = torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)
        dataset = CustomDataset(frames, None)
        dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers=0)
        # -- inference
        preds = inference(model, dataloader, device)
        inference_text = f'inference result: {str(preds[0])}'

        frames = []
        f = 1

    # writer.write(frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('i'):
        inference_status = True
        frames = []
    elif key == ord('q'):
        break

webcam.release()
# writer.release()
cv2.destroyAllWindows()