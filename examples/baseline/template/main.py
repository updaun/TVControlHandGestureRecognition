import os
import pandas as pd

import torch

from config import CFG
from seed import seed_everything
from model import BaseModel
from data import split_load_train_val, load_data
from train import train
from inference import inference

print(CFG)

# -- settings
seed_everything(CFG['SEED'])
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

DATASET_PATH = os.path.join(CFG['WORKING_DIR'], 'dataset/')
TRAIN_CSV_PATH = os.path.join(DATASET_PATH, 'train.csv')
TEST_CSV_PATH = os.path.join(DATASET_PATH, 'test.csv')
SUBMIT_PATH = os.path.join(CFG['WORKING_DIR'], 'submit/')

# -- data_loader
df = pd.read_csv(TRAIN_CSV_PATH)
df['path'] = df['path'].apply(lambda x : os.path.join(DATASET_PATH, x))
train_loader, val_loader = split_load_train_val(df, CFG['VAL_RATIO'])

test_df = pd.read_csv(TEST_CSV_PATH)
test_df['path'] = test_df['path'].apply(lambda x : os.path.join(DATASET_PATH, x))
test_loader = load_data(test_df, test_mode=True)

# -- model
model = BaseModel()

# -- train
best_model = train(model, train_loader, val_loader, device)

# -- inference
preds = inference(best_model, test_loader, device)

# -- submit
submit = pd.read_csv(os.path.join(DATASET_PATH, 'sample_submission.csv'))
submit['label'] = preds
submit.to_csv(os.path.join(SUBMIT_PATH, 'baseline_submit.csv'), index=False)