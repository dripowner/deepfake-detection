from clearml import Task, Logger, Dataset
from sklearn.model_selection import KFold
from itertools import repeat
import json
import os
from typing import Type
from pathlib import Path
from glob import glob
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import face_detector
from face_detector import VideoDataset
from face_detector import VideoFaceDetector
from utils import get_original_video_paths, get_original_with_fakes
from PIL import Image
from facenet_pytorch.models.mtcnn import MTCNN
import numpy as np
import pandas as pd
import random
# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


detector = MTCNN(margin=0, thresholds=[0.65, 0.75, 0.75], device="cpu")
cache = {}

def process_videos(videos, root_dir, detector_cls: Type[VideoFaceDetector]):
    detector = face_detector.__dict__[detector_cls](device="cuda:0")
    dataset = VideoDataset(videos)
    loader = DataLoader(dataset, shuffle=False, batch_size=1, collate_fn=lambda x: x)
    for item in tqdm(loader):
        result = {}
        video, indices, frames = item[0]
        batches = [frames[i:i + detector._batch_size] for i in range(0, len(frames), detector._batch_size)]
        for j, frames in enumerate(batches):
            result.update({int(j * detector._batch_size) + i : b for i, b in zip(indices, detector._detect_faces(frames))})
        id = os.path.splitext(os.path.basename(video))[0]
        out_dir = os.path.join(root_dir, "boxes")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "{}.json".format(id)), "w") as f:
            json.dump(result, f)


def extract_video(param, root_dir, crops_dir):
    video, bboxes_path = param
    with open(bboxes_path, "r") as bbox_f:
        bboxes_dict = json.load(bbox_f)

    capture = cv2.VideoCapture(video)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(frames_num):
        capture.grab()
        if i % 10 != 0:
            continue
        success, frame = capture.retrieve()
        if not success or str(i) not in bboxes_dict:
            continue
        id = os.path.splitext(os.path.basename(video))[0]
        crops = []
        bboxes = bboxes_dict[str(i)]
        if bboxes is None:
            continue
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
            w = xmax - xmin
            h = ymax - ymin
            p_h = h // 3
            p_w = w // 3
            crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
            h, w = crop.shape[:2]
            crops.append(crop)
        img_dir = os.path.join(root_dir, crops_dir, id)
        os.makedirs(img_dir, exist_ok=True)
        for j, crop in enumerate(crops):
            cv2.imwrite(os.path.join(img_dir, "{}_{}.png".format(i, j)), crop)


def get_video_paths(root_dir):
    paths = []
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        dir = Path(json_path).parent
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            original = v.get("original", None)
            if not original:
                original = k
            bboxes_path = os.path.join(root_dir, "boxes", original[:-4] + ".json")
            if not os.path.exists(bboxes_path):
                continue
            paths.append((os.path.join(dir, k), bboxes_path))

    return paths


def save_landmarks(ori_id, root_dir):
    ori_id = ori_id[:-4]
    ori_dir = os.path.join(root_dir, "crops", ori_id)
    landmark_dir = os.path.join(root_dir, "landmarks", ori_id)
    os.makedirs(landmark_dir, exist_ok=True)
    for frame in range(320):
        if frame % 10 != 0:
            continue
        for actor in range(2):
            image_id = "{}_{}.png".format(frame, actor)
            landmarks_id = "{}_{}".format(frame, actor)
            ori_path = os.path.join(ori_dir, image_id)
            landmark_path = os.path.join(landmark_dir, landmarks_id)

            if os.path.exists(ori_path):
                try:
                    image_ori = cv2.imread(ori_path, cv2.IMREAD_COLOR)[...,::-1]
                    frame_img = Image.fromarray(image_ori)
                    batch_boxes, conf, landmarks = detector.detect(frame_img, landmarks=True)
                    if landmarks is not None:
                        landmarks = np.around(landmarks[0].astype(np.float32))
                        np.save(landmark_path, landmarks)
                except Exception as e:
                    print(e)
                    pass


def save_diffs(pair, root_dir):
    ori_id, fake_id = pair
    ori_dir = os.path.join(root_dir, "crops", ori_id)
    fake_dir = os.path.join(root_dir, "crops", fake_id)
    diff_dir = os.path.join(root_dir, "diffs", fake_id)
    os.makedirs(diff_dir, exist_ok=True)
    for frame in range(320):
        if frame % 10 != 0:
            continue
        for actor in range(2):
            image_id = "{}_{}.png".format(frame, actor)
            diff_image_id = "{}_{}_diff.png".format(frame, actor)
            ori_path = os.path.join(ori_dir, image_id)
            fake_path = os.path.join(fake_dir, image_id)
            diff_path = os.path.join(diff_dir, diff_image_id)
            if os.path.exists(ori_path) and os.path.exists(fake_path):
                img1 = cv2.imread(ori_path, cv2.IMREAD_COLOR)
                img2 = cv2.imread(fake_path, cv2.IMREAD_COLOR)
                try:
                    # d, a = compare_ssim(img1, img2, multichannel=True, full=True)
                    d, a = structural_similarity(img1, img2, full=True, channel_axis=2)
                    a = 1 - a
                    diff = (a * 255).astype(np.uint8)
                    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(diff_path, diff)
                except:
                    pass


def get_paths(vid, label, root_dir):
    ori_vid, fake_vid = vid
    ori_dir = os.path.join(root_dir, "crops", ori_vid)
    fake_dir = os.path.join(root_dir, "crops", fake_vid)
    data = []
    for frame in range(320):
        if frame % 10 != 0:
            continue
        for actor in range(2):
            image_id = "{}_{}.png".format(frame, actor)
            ori_img_path = os.path.join(ori_dir, image_id)
            fake_img_path = os.path.join(fake_dir, image_id)
            img_path = ori_img_path if label == 0 else fake_img_path
            try:
                # img = cv2.imread(img_path)[..., ::-1]
                if os.path.exists(img_path):
                    data.append([img_path, label, ori_vid])
            except:
                pass
    return data


def unique_values(list): 
    result = [] 
    for sublist in list: 
        if sublist[0] not in result: 
            result.append(sublist[0]) 
    return result


# Arguments
args = {
    "project_name":"deepfake-detection",
    "train_dataset_name": "train-deepfake-detection",
    "train_dataset_version": "0.0.3",
    "n_splits": 4,
    "random_state": 42
}

task = Task.init(project_name="deepfake-detection", task_name="Preprocess training dataset")
logger = task.get_logger()
task.connect(args)

print('Get train dataset')
train_path = Dataset.get(
    dataset_name=args["train_dataset_name"], 
    dataset_project=args["project_name"],
    dataset_version=args["train_dataset_version"] 
).get_mutable_local_copy(Path("E:/deepfake-detection-dataset/train-deepfake-detection-0.4"))

print('Getting video paths')
originals = get_original_video_paths(train_path)

print("Delete fakes without originals")
with open(os.path.join(train_path, "data", "metadata.json")) as metadata_json:
    metadata = json.load(metadata_json)

keys_to_delete = []
for k, v in metadata.items():
    if v["label"] == "FAKE":
        if not os.path.exists(os.path.join(train_path, "data", v["original"])):
            os.remove(os.path.join(train_path, "data", k))
            keys_to_delete.append(k)

print("Delete original without fakes")
ori_with_fakes = unique_values(get_original_with_fakes(train_path)) # origs with fakes
all_origs = [x[:-4] for x in get_original_video_paths(train_path, basename=True)] # all origs
ori_to_delete = [x for x in all_origs if x not in ori_with_fakes]
for k, v in metadata.items():
    if v["label"] == "REAL" and k[:-4] in ori_to_delete:
        os.remove(os.path.join(train_path, "data", k))
        keys_to_delete.append(k)

print("Update metadata")
for k in keys_to_delete:
    del metadata[k]

with open(os.path.join(train_path, "data", "metadata.json"), "w") as f:
    json.dump(metadata, f)

print("Extracting bounding boxes from original videos")
process_videos(originals, train_path, "FacenetDetector")

print("Extracting crops as pngs in crops folder")
params = get_video_paths(train_path)
for item in tqdm(params):
    extract_video(item, train_path, "crops")

print("Extracting landmarks")
ids = get_original_video_paths(train_path, basename=True)
os.makedirs(os.path.join(train_path, "landmarks"), exist_ok=True)
for item in tqdm(ids):
    save_landmarks(item, train_path)

print("Extracting SSIM masks")
pairs = get_original_with_fakes(train_path)
os.makedirs(os.path.join(train_path, "diffs"), exist_ok=True)

for item in tqdm(pairs):
    save_diffs(item, train_path)

print("Generate folds")

ori_fakes = get_original_with_fakes(train_path)

folds = []
video_fold = {}
ori = get_original_video_paths(train_path, basename=True)

kf = KFold(n_splits=args["n_splits"], shuffle=True, random_state=args["random_state"])

for i, (train_index, test_index) in enumerate(kf.split(ori)):
    f = []
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")
    for item in test_index:
        f.append(ori[item])
    folds.append(f)
print(folds)
for d in os.listdir(train_path):
    if "data" in d:
        for f in os.listdir(os.path.join(train_path, d)):
            if "metadata.json" in f:
                with open(os.path.join(train_path, d, "metadata.json")) as metadata_json:
                    metadata = json.load(metadata_json)

                for k, v in metadata.items():
                    name = ""
                    if v["label"] == "FAKE":
                        name = v["original"]
                    else:
                        name = k
                    fold = None
                    print(name)
                    for i, fold_ori_names in enumerate(folds):
                        if name in fold_ori_names:
                            fold = i
                            break
                    assert fold is not None
                    video_id = k[:-4]
                    video_fold[video_id] = fold
for fold in range(len(folds)):
    holdoutset = {k for k, v in video_fold.items() if v == fold}
    trainset = {k for k, v in video_fold.items() if v != fold}
    assert holdoutset.isdisjoint(trainset), "Folds have leaks"
# sz = 50 // args["n_splits"]
# folds = []
# for fold in range(args["n_splits"]):
#     folds.append(list(range(sz * fold, sz * fold + sz if fold < args["n_splits"] - 1 else 50)))
# print(folds)
# video_fold = {}
# for d in os.listdir(train_path):
#     if "data" in d: # ????
#         part = int(d.split("_")[-1])
#         for f in os.listdir(os.path.join(train_path, d)):
#             if "metadata.json" in f:
#                 with open(os.path.join(train_path, d, "metadata.json")) as metadata_json:
#                     metadata = json.load(metadata_json)

#                 for k, v in metadata.items():
#                     fold = None
#                     for i, fold_dirs in enumerate(folds):
#                         if part in fold_dirs:
#                             fold = i
#                             break
#                     assert fold is not None
#                     video_id = k[:-4]
#                     video_fold[video_id] = fold
# for fold in range(len(folds)):
#     holdoutset = {k for k, v in video_fold.items() if v == fold}
#     trainset = {k for k, v in video_fold.items() if v != fold}
#     assert holdoutset.isdisjoint(trainset), "Folds have leaks"


# with Pool(processes=os.cpu_count()) as p:
#     with tqdm(total=len(ori_ori)) as pbar:
#         func = partial(get_paths, label=0, root_dir=args.root_dir)
#         for v in p.imap_unordered(func, ori_ori):
#             pbar.update()
#             data.extend(v)
#     with tqdm(total=len(ori_fakes)) as pbar:
#         func = partial(get_paths, label=1, root_dir=args.root_dir)
#         for v in p.imap_unordered(func, ori_fakes):
#             pbar.update()
#             data.extend(v)

data = []
ori_ori = set([(ori, ori) for ori, fake in ori_fakes])

with tqdm(total=len(ori_ori)) as pbar:
    for data_chunk in ori_ori:
        data.extend(get_paths(data_chunk, label=0, root_dir=train_path))
        pbar.update()

with tqdm(total=len(ori_fakes)) as pbar:
    for data_chunk in ori_fakes:
        data.extend(get_paths(data_chunk, label=1, root_dir=train_path))
        pbar.update()

fold_data = []
for img_path, label, ori_vid in data:
    path = Path(img_path)
    video = path.parent.name
    file = path.name
    assert video_fold[video] == video_fold[ori_vid], "original video and fake have leak  {} {}".format(ori_vid, video)
    fold_data.append([video, file, label, ori_vid, int(file.split("_")[0]), video_fold[video]])
random.shuffle(fold_data)
pd.DataFrame(fold_data, columns=["video", "file", "label", "original", "frame", "fold"]).to_csv(os.path.join(train_path, "folds.csv"), index=False)


print("Updating dataset")
train_dataset = Dataset.create(
    parent_datasets=["d8cd1ae64c45479395e70ecb95b0a98b"],
    dataset_name=args["train_dataset_name"],
    dataset_project=args["project_name"], 
    dataset_version='0.0.4'
)
train_dataset.add_files(path=train_path)
train_dataset.upload()
train_dataset.finalize()
