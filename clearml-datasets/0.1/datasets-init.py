import clearml
from clearml import Dataset


train_path = "E:/deepfake-detection-dataset/train-deepfake-detection-0.1"
val_path = "E:/deepfake-detection-dataset/val-deepfake-detection-0.1"
test_path = "E:/deepfake-detection-dataset/test-deepfake-detection-0.1"

train_dataset = Dataset.create(
    dataset_name="train-deepfake-detection-0.1", 
    dataset_project="deepfake-detection"
)
train_dataset.add_files(path=train_path)

val_dataset = Dataset.create(
    dataset_name="val-deepfake-detection-0.1", 
    dataset_project="deepfake-detection"
)
val_dataset.add_files(path=val_path)

test_dataset = Dataset.create(
    dataset_name="test-deepfake-detection-0.1", 
    dataset_project="deepfake-detection"
)
test_dataset.add_files(path=test_path)

train_dataset.upload()
val_dataset.upload()
test_dataset.upload()

train_dataset.finalize()
val_dataset.finalize()
test_dataset.finalize()