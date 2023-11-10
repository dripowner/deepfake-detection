from clearml import Dataset
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from shutil import copy2


path = "E:/deepfake-detection-dataset/deepfake-detection-0.1"

metadata_df = pd.read_json(os.path.join(path, "metadata.json")).T
metadata_df.reset_index(inplace=True)
metadata_df.columns = ["filename", "label", "split", "original"]

print(len(metadata_df))

X = metadata_df.drop(axis=1, columns=["label"])
y = metadata_df.label

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, stratify=y, random_state=42)

train_df = pd.concat([x_train, y_train], axis=1)
train_df.set_index("filename", inplace=True)
print(len(train_df))

test_df = pd.concat([x_test, y_test], axis=1)
test_df.set_index("filename", inplace=True)
print(len(test_df))



train_path = "E:/deepfake-detection-dataset/train-deepfake-detection-0.3"
test_path = "E:/deepfake-detection-dataset/test-deepfake-detection-0.3"

if not os.path.isdir(test_path):
  os.mkdir(test_path)

if not os.path.isdir(train_path) or not os.path.isdir(os.path.join(train_path, "data")):
  os.mkdir(train_path)
  os.mkdir(os.path.join(train_path, "data"))

train_df.to_json(os.path.join(train_path, "data", "metadata.json"), orient="index")
test_df.to_json(os.path.join(test_path, "metadata.json"), orient="index")

for index, row in train_df.iterrows():
  print(row)
  if not os.path.exists(os.path.join(train_path, "data", index)):
    copy2(os.path.join(path, index), os.path.join(train_path, "data", index))
  else:
    continue

for index, row in test_df.iterrows():
  print(row)
  if not os.path.exists(os.path.join(test_path, index)):
    copy2(os.path.join(path, index), os.path.join(test_path, index))
  else:
    continue

test_dataset = Dataset.create(
  dataset_name='test-deepfake-detection',
  dataset_project="deepfake-detection", 
  dataset_version='0.0.3'
)
test_dataset.add_files(path=test_path)
test_dataset.upload()
test_dataset.finalize()


train_dataset = Dataset.create(
  dataset_name='train-deepfake-detection',
  dataset_project="deepfake-detection", 
  dataset_version='0.0.3'
)
train_dataset.add_files(path=train_path)
train_dataset.upload()
train_dataset.finalize()

