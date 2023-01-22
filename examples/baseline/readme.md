# notebook/baseline.ipynb
- baseline 코드 구축
# template
- baseline 코드 템플릿화

- train 및 inference
```
python main.py
```

- 저장된 모델 pth 파일을 통한 inference
```
python inference.py
```

## config.py

- 'MODEL': baseline, resnet3d18
- 'FPS':30,
- 'IMG_SIZE'
- 'EPOCHS'
- 'LEARNING_RATE'
- 'BATCH_SIZE'
- 'SEED'
- 'VAL_RATIO'
- 'WORKING_DIR'
- 'WEIGHTS': *.pth