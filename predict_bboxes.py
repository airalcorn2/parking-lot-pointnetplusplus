import importlib
import numpy as np
import os
import pickle
import sys
import torch
import wandb

from env_vars import *
from parking_lot_dataset import ParkingLotTestDataset


def load_wandb_model(run_id, device):
    api = wandb.Api()
    # See: https://github.com/wandb/wandb/issues/3678.
    run_path = f"{ENTITY}/{WANDB_PROJECT}/{run_id}"
    run = api.run(run_path)
    config = run.config
    root = f"{WANDB_RUNS}/{run_id}"
    os.makedirs(root, exist_ok=True)
    with open(f"{root}/{CONFIG_F}", "wb") as f:
        pickle.dump(config, f)

    _ = wandb.restore(MODEL_F, run_path, replace=True, root=root)
    module = MODEL_F.split(".")[0]
    model_cls = importlib.import_module(".".join(root.split("/")) + f".{module}")
    model = model_cls.PointNetPlusPlus(**config["model_args"]).to(device)

    # See: https://wandb.ai/lavanyashukla/save_and_restore/reports/Saving-and-Restoring-Machine-Learning-Models-with-W-B--Vmlldzo3MDQ3Mw
    # and: https://docs.wandb.ai/guides/track/save-restore.
    weights_f = wandb.restore(BEST_PARAMS_F, run_path, replace=True, root=root)
    model.load_state_dict(torch.load(weights_f.name))
    model.eval()

    return (model, config)


def main():
    device = torch.device("cuda:0")
    run_id = sys.argv[1]
    (model, config) = load_wandb_model(run_id, device)
    scale = np.array(config["scale"])
    shift = np.array(config["shift"])

    npy_fs = os.listdir(f"{DOCKER_TEST_PATH}/{HUMANS_FOLDER}")
    dataset_args = {
        "data_dir": DOCKER_TEST_PATH,
        "npy_fs": npy_fs,
        "max_points": config["max_points"],
    }
    test_dataset = ParkingLotTestDataset(**dataset_args)

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            npy_f = test_dataset.npy_fs[idx]

            xyzs, feats = test_dataset[idx]
            preds = model(xyzs[None].to(device), feats[None].to(device))[0].cpu()

            preds[:-2] = (preds[:-2] + 1) / scale + shift

            extent = torch.zeros(3)
            extent[:2] = preds[2:4]
            human_height = np.load(f"{DOCKER_TEST_PATH}/{HEIGHTS_FOLDER}/{npy_f}")
            extent[2] = human_height.item() + preds[4]

            center = torch.ones(4)
            xy_center_offset = preds[:2]
            xyzs = np.load(f"{DOCKER_TEST_PATH}/{HUMANS_FOLDER}/{npy_f}")
            xy_medians = torch.Tensor(np.median(xyzs[:, :2], 0))
            center[:2] = xy_medians + xy_center_offset

            # The human point cloud has the feet cut off.
            diff = human_height.item() - xyzs[:, 2].max()
            center[2] = extent[2] / 2 - diff
            inv_R_t_path = f"{DOCKER_TEST_PATH}/{TRANSFORMATIONS_FOLDER}/{npy_f}"
            inv_R_t = torch.Tensor(np.load(inv_R_t_path))
            center = inv_R_t @ center
            center = center[:3]

            c = preds[5]
            sign = torch.sign(preds[6])
            s = sign * (1 - c**2) ** 0.5
            bbox_R = torch.Tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]]).flatten()

            label = torch.cat([center, extent, bbox_R])
            parts = npy_f.split(".")[0].split("_")
            npy_f = f"{parts[0]}_{parts[-1]}.npy"
            np.save(f"{DOCKER_TEST_PATH}/{LABELS_FOLDER}/{npy_f}", label.cpu().numpy())


if __name__ == "__main__":
    main()
