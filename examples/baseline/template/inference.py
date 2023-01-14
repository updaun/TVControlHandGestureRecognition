from tqdm.auto import tqdm

import torch

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

if __name__ == "__main__":
    import os
    import pandas as pd

    import torch.nn as nn
    from torchvision import models

    from data import load_data
    from model import BaseModel
    from config import CFG

    # -- settings
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    DATASET_PATH = os.path.join(CFG['WORKING_DIR'], 'dataset/')
    TEST_CSV_PATH = os.path.join(DATASET_PATH, 'test.csv')
    SUBMIT_PATH = os.path.join(CFG['WORKING_DIR'], 'submit/')
    
    # -- data_loader
    test_df = pd.read_csv(TEST_CSV_PATH)
    test_df['path'] = test_df['path'].apply(lambda x : os.path.join(DATASET_PATH, x))
    test_loader = load_data(test_df, test_mode=True)

    # -- model
    if CFG['MODEL'] == 'baseline':
        model = BaseModel()
    elif CFG['MODEL'] == 'resnet3d18':
        # model = models.video.r3d_18(pretrained=True)
        model = models.video.r3d_18(weights='R3D_18_Weights.DEFAULT')
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 5)
    weights_path = os.path.join(f'./weights/{CFG["MODEL"]}', CFG['WEIGHTS'])
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # -- inference
    preds = inference(model, test_loader, device)

    # -- submit
    submit = pd.read_csv(os.path.join(DATASET_PATH, 'sample_submission.csv'))
    submit['label'] = preds
    submit.to_csv(os.path.join(SUBMIT_PATH, f'{CFG["MODEL"]}_submit.csv'), index=False)