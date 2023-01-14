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
    model = BaseModel()
    weights_path = os.path.join('./weights', CFG['WEIGHTS'])
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # -- inference
    preds = inference(model, test_loader, device)

    # -- submit
    submit = pd.read_csv(os.path.join(DATASET_PATH, 'sample_submission.csv'))
    submit['label'] = preds
    submit.to_csv(os.path.join(SUBMIT_PATH, 'submit.csv'), index=False)