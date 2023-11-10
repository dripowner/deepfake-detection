from clearml import Task, OutputModel, Dataset
import json
import os
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import log_loss
from torch import topk
import re
import glob
import losses
from classifier_dataset import DeepFakeClassifierDataset
from losses import WeightedLosses
from config import load_config
from utils import create_optimizer, AverageMeter
import classifiers

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
import numpy as np

from tensorboardX import SummaryWriter

import torch
from torch.backends import cudnn
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur
from albu import IsotropicResize
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.backends.cudnn.benchmark = True


def clean_folder(folder_path):
    # get a list of all file paths in the folder
    file_paths = glob.glob(os.path.join(folder_path, '*'))

    # iterate over each file path and delete the file
    for file_path in file_paths:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def create_train_transforms(size=300):
    return Compose([
        ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        GaussNoise(p=0.1),
        GaussianBlur(blur_limit=3, p=0.05),
        HorizontalFlip(),
        OneOf([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
        ], p=1),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
        ToGray(p=0.2),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    ]
    )


def create_val_transforms(size=300):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])


def main():

    args = {
        'config': "./configs/config.json",
        'gpu': '0',
        'output-dir': './weights/',
        'resume': '',
        'fold': 0,
        'prefix': 'classifier_',
        'data-dir': "E:/deepfake-detection-dataset/train-deepfake-detection-0.4",
        'folds-csv': 'folds.csv',
        'crops-dir': 'crops',
        'label-smoothing': 0.01,
        'logdir': 'logs',
        'freeze-epochs': 0,
        'local_rank': 0,
        'seed': 777,
        'test_every': 1,
        'only-changed-frames': False,
        "project_name":"deepfake-detection",
        "train_dataset_name": "train-deepfake-detection",
        "train_dataset_version": "0.0.4",
        "from-zero": True,
        "zero-score": False
    }

    task = Task.init(project_name="deepfake-detection", task_name="Training model")
    output_model = OutputModel(task=task)
    logger = task.get_logger()
    task.connect(args)

    clean_folder(args["logdir"])
    clean_folder(args["output-dir"])

    print('Get train dataset')
    train_path = Dataset.get(
        dataset_name=args["train_dataset_name"], 
        dataset_project=args["project_name"],
        dataset_version=args["train_dataset_version"] 
    ).get_local_copy()

    args["data-dir"] = train_path
    args["folds-csv"] = os.path.join(args["data-dir"], "folds.csv")

    os.makedirs(args["output-dir"], exist_ok=True)


    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu"]

    cudnn.benchmark = True

    conf = load_config(args["config"])
    task.upload_artifact('training config', artifact_object=conf)
    
    model = classifiers.__dict__[conf['network']](encoder=conf['encoder'])

    model = model.cuda()


    reduction = "mean"
    loss_fn = []
    weights = []
    for loss_name, weight in conf["losses"].items():
        loss_fn.append(losses.__dict__[loss_name](reduction=reduction).cuda())
        weights.append(weight)
    loss = WeightedLosses(loss_fn, weights)
    loss_functions = {"classifier_loss": loss}
    optimizer, scheduler = create_optimizer(conf['optimizer'], model)
    bce_best = 100
    start_epoch = 0
    batch_size = conf['optimizer']['batch_size']

    data_train = DeepFakeClassifierDataset(mode="train",
                                           fold=args["fold"],
                                           crops_dir=args["crops-dir"],
                                           data_path=args["data-dir"],
                                           label_smoothing=args["label-smoothing"],
                                           folds_csv=args["folds-csv"],
                                           normalize=conf.get("normalize", None),
                                           transforms=create_train_transforms(conf["size"]))
    data_val = DeepFakeClassifierDataset(mode="val",
                                         fold=args["fold"],
                                         crops_dir=args["crops-dir"],
                                         data_path=args["data-dir"],
                                         folds_csv=args["folds-csv"],
                                         normalize=conf.get("normalize", None),
                                         transforms=create_val_transforms(conf["size"]))
    val_data_loader = DataLoader(data_val, batch_size=batch_size * 2, shuffle=False, pin_memory=False)
    os.makedirs(args["logdir"], exist_ok=True)
    summary_writer = SummaryWriter(args["logdir"] + '/' + conf.get("prefix", args["prefix"]) + conf['encoder'] + "_" + str(args["fold"]))
    if args["resume"]:
        if os.path.isfile(args["resume"]):
            print("=> loading checkpoint '{}'".format(args["resume"]))
            checkpoint = torch.load(args["resume"], map_location='cpu')
            state_dict = checkpoint['state_dict']
            state_dict = {k[7:]: w for k, w in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            if not args["from-zero"]:
                start_epoch = checkpoint['epoch']
                if not args["zero-score"]:
                    bce_best = checkpoint.get('bce_best', 0)
            print("=> loaded checkpoint '{}' (epoch {}, bce_best {})"
                  .format(args["resume"], checkpoint['epoch'], checkpoint['bce_best']))
        else:
            print("=> no checkpoint found at '{}'".format(args["resume"]))
    if args["from-zero"]:
        start_epoch = 0
    current_epoch = start_epoch


    snapshot_name = "{}{}_{}_{}".format(conf.get("prefix", args["prefix"]), conf['network'], conf['encoder'], args["fold"])

    data_val.reset(1, args["seed"])
    max_epochs = conf['optimizer']['schedule']['epochs']
    for epoch in range(start_epoch, max_epochs):
        data_train.reset(epoch, args["seed"])
        train_sampler = None
        if epoch < args["freeze-epochs"]:
            print("Freezing encoder!!!")
            model.encoder.eval()
            for p in model.encoder.parameters():
                p.requires_grad = False
        else:
            model.encoder.train()
            for p in model.encoder.parameters():
                p.requires_grad = True

        train_data_loader = DataLoader(data_train, batch_size=batch_size,
                                       shuffle=train_sampler is None, sampler=train_sampler, pin_memory=False,
                                       drop_last=True)

        train_epoch(current_epoch, loss_functions, model, optimizer, scheduler, train_data_loader, summary_writer, conf,
                    args["local_rank"], args["only-changed-frames"])
        model = model.eval()

        if args["local_rank"] == 0:
            torch.save({
                'epoch': current_epoch + 1,
                'state_dict': model.state_dict(),
                'bce_best': bce_best,
            }, args["output-dir"] + '/' + snapshot_name + "_last")
            torch.save({
                'epoch': current_epoch + 1,
                'state_dict': model.state_dict(),
                'bce_best': bce_best,
            }, args["output-dir"] + snapshot_name + "_{}".format(current_epoch))
            if (epoch + 1) % args["test_every"] == 0:
                bce_best = evaluate_val(args, val_data_loader, bce_best, model,
                                        snapshot_name=snapshot_name,
                                        current_epoch=current_epoch,
                                        summary_writer=summary_writer)
        current_epoch += 1

    # pattern = re.compile(f'.*{re.escape("best_dice")}.*')

    # for filename in os.listdir("../weights"):
    #     if pattern.match(filename):
    #         model_url = filename
    # output_model.update_weights(register_uri=model_url)


def evaluate_val(args, data_val, bce_best, model, snapshot_name, current_epoch, summary_writer):
    print("Test phase")
    model = model.eval()

    bce, probs, targets = validate(model, data_loader=data_val)
    if args["local_rank"] == 0:
        summary_writer.add_scalar('val/bce', float(bce), global_step=current_epoch)
        if bce < bce_best:
            print("Epoch {} improved from {} to {}".format(current_epoch, bce_best, bce))
            if args["output-dir"] is not None:
                torch.save({
                    'epoch': current_epoch + 1,
                    'state_dict': model.state_dict(),
                    'bce_best': bce,
                }, args["output-dir"] + snapshot_name + "_best_dice")
            bce_best = bce
            with open("predictions_{}.json".format(args["fold"]), "w") as f:
                json.dump({"probs": probs, "targets": targets}, f)
        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'bce_best': bce_best,
        }, args["output-dir"] + snapshot_name + "_last")
        print("Epoch: {} bce: {}, bce_best: {}".format(current_epoch, bce, bce_best))
    return bce_best


def validate(net, data_loader, prefix=""):
    probs = defaultdict(list)
    targets = defaultdict(list)

    with torch.no_grad():
        for sample in tqdm(data_loader):
            imgs = sample["image"].cuda()
            img_names = sample["img_name"]
            labels = sample["labels"].cuda().float()
            out = net(imgs)
            labels = labels.cpu().numpy()
            preds = torch.sigmoid(out).cpu().numpy()
            for i in range(out.shape[0]):
                video, img_id = img_names[i].split("\\")
                probs[video].append(preds[i].tolist())
                targets[video].append(labels[i].tolist())
    data_x = []
    data_y = []
    for vid, score in probs.items():
        score = np.array(score)
        lbl = targets[vid]

        score = np.mean(score)
        lbl = np.mean(lbl)
        data_x.append(score)
        data_y.append(lbl)
    y = np.array(data_y)
    x = np.array(data_x)
    fake_idx = y > 0.1
    real_idx = y < 0.1
    fake_loss = log_loss(y[fake_idx], x[fake_idx], labels=[0, 1])
    real_loss = log_loss(y[real_idx], x[real_idx], labels=[0, 1])
    print("{}fake_loss".format(prefix), fake_loss)
    print("{}real_loss".format(prefix), real_loss)

    return (fake_loss + real_loss) / 2, probs, targets


def train_epoch(current_epoch, loss_functions, model, optimizer, scheduler, train_data_loader, summary_writer, conf,
                local_rank, only_valid):
    losses = AverageMeter()
    fake_losses = AverageMeter()
    real_losses = AverageMeter()
    max_iters = conf["batches_per_epoch"]
    print("training epoch {}".format(current_epoch))
    model.train()
    pbar = tqdm(enumerate(train_data_loader), total=max_iters, desc="Epoch {}".format(current_epoch), ncols=0)
    if conf["optimizer"]["schedule"]["mode"] == "epoch":
        scheduler.step(current_epoch)
    for i, sample in pbar:
        imgs = sample["image"].cuda()
        labels = sample["labels"].cuda().float()
        out_labels = model(imgs)
        if only_valid:
            valid_idx = sample["valid"].cuda().float() > 0
            out_labels = out_labels[valid_idx]
            labels = labels[valid_idx]
            if labels.size(0) == 0:
                continue

        fake_loss = 0
        real_loss = 0
        fake_idx = labels > 0.5
        real_idx = labels <= 0.5


        if torch.sum(fake_idx * 1) > 0:
            fake_loss = loss_functions["classifier_loss"](out_labels[fake_idx], labels[fake_idx])
        if torch.sum(real_idx * 1) > 0:
            real_loss = loss_functions["classifier_loss"](out_labels[real_idx], labels[real_idx])

        loss = (fake_loss + real_loss) / 2
        losses.update(loss.item(), imgs.size(0))
        fake_losses.update(0 if fake_loss == 0 else fake_loss.item(), imgs.size(0))
        real_losses.update(0 if real_loss == 0 else real_loss.item(), imgs.size(0))

        optimizer.zero_grad()
        pbar.set_postfix({"lr": float(scheduler.get_lr()[-1]), "epoch": current_epoch, "loss": losses.avg,
                          "fake_loss": fake_losses.avg, "real_loss": real_losses.avg})


        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        if conf["optimizer"]["schedule"]["mode"] in ("step", "poly"):
            scheduler.step(i + current_epoch * max_iters)
        if i == max_iters - 1:
            break
    pbar.close()
    if local_rank == 0:
        for idx, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            summary_writer.add_scalar('group{}/lr'.format(idx), float(lr), global_step=current_epoch)
        summary_writer.add_scalar('train/loss', float(losses.avg), global_step=current_epoch)


if __name__ == '__main__':
    main()