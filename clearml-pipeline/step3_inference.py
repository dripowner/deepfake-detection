from clearml import Task, InputModel, Logger, Dataset
import os
import re
import time
import torch
import pandas as pd
from kernel_utils import VideoReader, FaceExtractor, confident_strategy, predict_on_video_set
from classifiers import DeepFakeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
import logging


# Arguments
args = {
    "project_name":"deepfake-detection",
    "model_id": "b5f3986b6dd14ae7a63762b0fd4d52e0",
    "test_dataset_name": "test-deepfake-detection",
    "test_dataset_version": "0.0.3"
}

input_model = InputModel(model_id=args["model_id"])

# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name="deepfake-detection", task_name="Pretrained model inference")

logger = task.get_logger()

task.connect(args)
task.connect(input_model)

print('Get Test dataset')
test_path = Dataset.get(
    dataset_name=args["test_dataset_name"], 
    dataset_project=args["project_name"],
    dataset_version=args["test_dataset_version"] 
).get_local_copy()
print('Test dataset loaded')

models = []
model_paths = [input_model.get_local_copy()]

for path in model_paths:
    model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to("cuda")
    print("loading state dict {}".format(path))
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=True)
    model.eval()
    del checkpoint
    models.append(model.half())

frames_per_video = 32
video_reader = VideoReader()
video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn)
input_size = 380
strategy = confident_strategy
stime = time.time()

test_videos = sorted([x for x in os.listdir(test_path) if x[-4:] == ".mp4"])
print("Predicting {} videos".format(len(test_videos)))
predictions = predict_on_video_set(face_extractor=face_extractor, input_size=input_size, models=models,
                                       strategy=strategy, frames_per_video=frames_per_video, videos=test_videos,
                                       num_workers=6, test_dir=test_path)

inf_time = time.time() - stime
print("Elapsed:", inf_time)

submission_df = pd.DataFrame({"filename": test_videos, "predicted_label": predictions})

df = pd.read_json(os.path.join(test_path, "metadata.json")).T
df.drop(axis=1, columns=["split", "original"], inplace=True)
df.reset_index(inplace=True)
df.columns = ["filename", "label"]
df = df.merge(submission_df, on="filename")
df.columns=["filename", "label", "predicted_label"]
df["probs"] = df["predicted_label"].apply(lambda x: [round(x, 3), round(1 - x, 3)])
df["predicted_label"] = df["predicted_label"].apply(lambda x: "FAKE" if x >= 0.4 else "REAL")

task.upload_artifact('predicted labels', artifact_object=df)

logger.report_table(title="Classification Report", 
                    table_plot=pd.DataFrame.from_dict(classification_report(df["label"], df["predicted_label"], output_dict=True), orient='columns').T,
                    iteration=0, 
                    series='pandas DataFrame')
logger.report_single_value("Log Loss", log_loss(list(df["label"]), list(df["probs"]), labels=["REAL", "FAKE"]))
logger.report_confusion_matrix("Confusion Matrix", "ignored", 
                               iteration=0, 
                               matrix=confusion_matrix(df["label"], df["predicted_label"], labels=["FAKE", "REAL"]),
                               xaxis="Predicted", yaxis="True")

logger.report_single_value("Inference Time", inf_time)
logger.report_single_value("Mean inference time for 1 video", inf_time / len(test_videos))

