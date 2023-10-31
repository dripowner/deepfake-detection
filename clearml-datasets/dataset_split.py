import clearml
from clearml import Dataset
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from shutil import copy2


path = "E:/deepfake-detection-dataset/deepfake-detection-0.1"

metadata_df = pd.read_json(os.path.join(path, "metadata.json")).T
metadata_df.drop(axis=1, columns=["original", "split"], inplace=True)
metadata_df.reset_index(inplace=True)
metadata_df.columns = ["filename", "label"]

X = metadata_df.drop(axis=1, columns=["label"])
y = metadata_df.label

x_t, x_test, y_t, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x_t, y_t, test_size = 0.25, train_size=0.75, stratify=y_t)

train_df = pd.concat([x_train, y_train], axis=1)
train_df["split"] = "train"
train_df.set_index("filename", inplace=True)

test_df = pd.concat([x_test, y_test], axis=1)
test_df["split"] = "test"
test_df.set_index("filename", inplace=True)

val_df = pd.concat([x_val, y_val], axis=1)
val_df["split"] = "val"
val_df.set_index("filename", inplace=True)

train_path = "E:/deepfake-detection-dataset/train-deepfake-detection-0.1"
val_path = "E:/deepfake-detection-dataset/val-deepfake-detection-0.1"
test_path = "E:/deepfake-detection-dataset/test-deepfake-detection-0.1"

train_df.to_json(os.path.join(train_path, "metadata.json"), orient="index")
val_df.to_json(os.path.join(val_path, "metadata.json"), orient="index")
test_df.to_json(os.path.join(test_path, "metadata.json"), orient="index")

res_df = pd.concat([train_df, val_df, test_df])

for index, row in res_df.iterrows():
    print(row['split'], row['label'])
    if row["split"] == "train":
        copy2(os.path.join(path, index), os.path.join(train_path, index))
    elif row["split"] == "val":
        copy2(os.path.join(path, index), os.path.join(val_path, index))
    else:
        copy2(os.path.join(path, index), os.path.join(test_path, index))
