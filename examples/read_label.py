import pandas as pd


train_csv = "dataset/train.csv"

csv = pd.read_csv(train_csv)
print(csv["label"].to_list())