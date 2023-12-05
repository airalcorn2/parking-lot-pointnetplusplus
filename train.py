import json
import numpy as np
import os
import random
import shutil
import torch
import wandb

from config import config
from env_vars import *
from parking_lot_dataset import ParkingLotDataset
from pointnetplusplus import PointNetPlusPlus
from torch import nn, optim
from torch.utils.data import DataLoader


def main():
    label_fs = os.listdir(f"{DOCKER_TRAIN_PATH}/{LABELS_FOLDER}")
    num_points = []
    labels = []
    for label_f in label_fs:
        num_points.append(
            len(np.load(f"{DOCKER_TRAIN_PATH}/{HUMANS_FOLDER}/{label_f}"))
        )
        labels.append(np.load(f"{DOCKER_TRAIN_PATH}/{LABELS_FOLDER}/{label_f}"))

    max_points = max(num_points)
    labels = np.stack(labels)
    shift = labels[:, :-2].min(0)
    labels[:, :-2] = labels[:, :-2] - shift
    scale = 2 / labels[:, :-2].max(0)
    config["max_points"] = max_points
    config["shift"] = shift
    config["scale"] = scale
    config["model_args"]["out_feats"] = labels.shape[1]

    try:
        with open(DATA_DICT_F) as f:
            data_dict = json.load(f)

    except FileNotFoundError:
        label_fs = os.listdir(f"{DOCKER_TRAIN_PATH}/{LABELS_FOLDER}")
        random.shuffle(label_fs)

        train_p = config["train_p"]
        train_n = int(train_p * len(label_fs))
        train_label_fs = label_fs[:train_n]
        valid_label_fs = label_fs[train_n:]
        data_dict = {"train": train_label_fs, "valid": valid_label_fs}
        with open(DATA_DICT_F, "w") as f:
            json.dump(data_dict, f)

    dataset_args = {
        "data_dir": DOCKER_TRAIN_PATH,
        "label_fs": data_dict["train"],
        "max_points": max_points,
        "shift": shift,
        "scale": scale,
    }
    train_dataset = ParkingLotDataset(**dataset_args)
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    train_loader = DataLoader(train_dataset, batch_size, True, num_workers=num_workers)
    dataset_args["label_fs"] = data_dict["valid"]
    valid_dataset = ParkingLotDataset(**dataset_args)
    valid_loader = DataLoader(valid_dataset, batch_size, num_workers=num_workers)

    config["model_args"]["in_feats"] = train_dataset[0][1].shape[1]

    config["data_dict"] = data_dict
    wandb.init(project=WANDB_PROJECT, entity=ENTITY, config=config)
    shutil.copyfile(MODEL_F, f"{wandb.run.dir}/{MODEL_F}")

    device = torch.device("cuda:0")
    model = PointNetPlusPlus(**config["model_args"]).to(device)
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params}")

    lr = 1e-3
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction="sum")

    best_train_loss = float("inf")
    best_valid_loss = float("inf")
    patience = config["patience"]
    max_lr_drops = config["max_lr_drops"]
    lr_reducer = config["lr_reducer"]
    no_improvement = 0
    lr_drops = 0
    lr_reductions = 0
    total_train_loss = float("inf")
    for epoch in range(config["epochs"]):
        model.eval()
        total_valid_loss = 0.0
        with torch.no_grad():
            for xyzs, feats, tgts in valid_loader:
                preds = model(xyzs.to(device), feats.to(device))
                loss = criterion(preds, tgts.to(device))
                total_valid_loss += loss.item()

        if total_valid_loss < best_valid_loss:
            best_valid_loss = total_valid_loss
            no_improvement = 0
            lr_drops = 0
            torch.save(model.state_dict(), f"{wandb.run.dir}/{BEST_PARAMS_F}")

        else:
            no_improvement += 1
            if no_improvement == patience:
                lr_drops += 1
                if lr_drops == max_lr_drops:
                    break

                no_improvement = 0
                lr_reductions += 1
                for g in optimizer.param_groups:
                    g["lr"] *= lr_reducer

        wandb.log(
            {
                "average_train_loss": total_train_loss / len(train_dataset),
                "average_valid_loss": total_valid_loss / len(valid_dataset),
                "lr_reductions": lr_reductions,
            }
        )

        model.train()
        total_train_loss = 0.0
        for xyzs, feats, tgts in train_loader:
            optimizer.zero_grad()
            preds = model(xyzs.to(device), feats.to(device))
            loss = criterion(preds, tgts.to(device))
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        if total_train_loss < best_train_loss:
            best_train_loss = total_train_loss


if __name__ == "__main__":
    main()
