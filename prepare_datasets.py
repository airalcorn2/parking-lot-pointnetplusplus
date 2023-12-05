import boto3
import json
import multiprocessing
import numpy as np
import open3d as o3d
import os
import sys

from env_vars import *
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors


def level_scene(pcd):
    points = np.array(pcd.points)

    # Create a grid of points.
    xs = np.linspace(0, MAX_X, N_GRID_POINTS)
    ys = np.linspace(-MAX_Y, MAX_Y, N_GRID_POINTS)
    grid = np.stack(np.meshgrid(xs, ys)).transpose(1, 2, 0)
    grid_points = grid.reshape(-1, 2)
    grid_points = np.hstack([grid_points, np.zeros(len(grid_points))[:, None]])
    grid_points += np.array([0, 0, points[:, 2].min()])

    # For each grid point, find nearest neighbor in scene point cloud. These are our
    # "ground" points.
    nbrs = NearestNeighbors(n_neighbors=1).fit(points)
    (_, pcd_idxs) = nbrs.kneighbors(grid_points)
    pcd_idxs = np.unique(pcd_idxs)
    ground_points = points[pcd_idxs]

    # Estimate the plane for the ground points.
    X = np.hstack([ground_points[:, :2], np.ones(len(ground_points))[:, None]])
    y = ground_points[:, 2]
    coefs = np.linalg.lstsq(X, y, rcond=None)[0]

    # Calculate the coordinate rotation matrix for the ground plane.
    # See: https://math.stackexchange.com/a/476311/614328.
    z_axis = -np.array([coefs[0], coefs[1], -1])
    a = z_axis / np.linalg.norm(z_axis)
    b = np.array([0, 0, 1])
    v = np.cross(a, b)
    skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + skew + skew @ skew * 1 / (1 + a[2])

    return R


def extract_human_points(pcd):
    points = np.array(pcd.points)
    keep_y = np.abs(points[:, 1]) < Y_THRESH
    keep_x = (X_MIN < points[:, 0]) & (points[:, 0] < X_MAX)
    keep_points = points[keep_x & keep_y]

    R = level_scene(pcd)
    new_points = (R @ keep_points.T).T
    above_ground = new_points[:, 2] > Z_THRESH
    human_points = new_points[above_ground]
    # No human present.
    if len(human_points) == 0:
        return None

    ground = np.median(new_points[~above_ground, 2])
    human_height = new_points[:, 2].max() - ground

    t = -np.median(human_points, 0)
    human_points = human_points + t
    dists = np.linalg.norm(human_points, axis=1)
    dists.sort()
    # Take the fifth largest distance to avoid outliers.
    max_dist = dists[-5] + BUFFER
    human_points = new_points + t
    dists = np.linalg.norm(human_points, axis=1)
    human_points = human_points[dists < max_dist]

    R_inv = R.T
    t_inv = R_inv @ (-t)
    inv_R_t = np.hstack([R_inv, t_inv[:, None]])

    return {
        "keep_points": keep_points,
        "new_points": new_points,
        "human_points": human_points,
        "inv_R_t": inv_R_t,
        "t": t,
        "ground": ground,
        "human_height": human_height,
    }


def get_train_sample(sample_dict, bucket, visualize):
    frames = sample_dict["attributes"]["frames"]
    frame_annotations = sample_dict["labels"]["ground-truth"]["attributes"]["frames"]
    all_frame_annotations = list(zip(frames, frame_annotations))
    for frame, annotations in all_frame_annotations:
        if len(annotations["annotations"]) == 0:
            continue

        name = frame["name"]
        if os.path.exists(f"{TEST_PATH}/{HUMANS_FOLDER}/{name}.npy"):
            os.remove(f"{TEST_PATH}/{HUMANS_FOLDER}/{name}.npy")
            os.remove(f"{TEST_PATH}/{TRANSFORMATIONS_FOLDER}/{name}.npy")
            os.remove(f"{TEST_PATH}/{HEIGHTS_FOLDER}/{name}.npy")

        if os.path.exists(f"{TRAIN_PATH}/{HUMANS_FOLDER}/{name}_True.npy"):
            continue

        pcd_path = f"{DATASETS_PATH}/{PCDS_FOLDER}/{name}.pcd"
        if not os.path.exists(pcd_path):
            bucket.download_file(f"{S3_FOLDER}/{name}.pcd", pcd_path)

        labels = []
        for bbox in annotations["annotations"]:
            center = []
            extent = []
            for xyz in ["x", "y", "z"]:
                center.append(bbox["position"][xyz])
                extent.append(bbox["dimensions"][xyz])

            center = np.array(center)
            extent = np.array(extent)

            q = []
            for xyzw in ["x", "y", "z", "w"]:
                q.append(bbox["rotation"][f"q{xyzw}"])

            q = np.array(q)
            labels.append(np.concatenate([center, extent, q]))

        labels = np.concatenate(labels)
        center = labels[:3]
        extent = labels[3:6]
        q = labels[6:10]

        pcd = o3d.io.read_point_cloud(pcd_path)
        bbox_R = Rotation.from_quat(q).as_matrix()
        bbox = o3d.geometry.OrientedBoundingBox(center, bbox_R, extent)

        for mirror in [False, True]:
            samp_name = f"{name}_{mirror}"
            if mirror:
                points = np.array(pcd.points)
                points[:, 1] = -points[:, 1]
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

                center[1] = -center[1]
                rotvec = Rotation.from_quat(q).as_rotvec()
                rotvec[-1] = -rotvec[-1]
                bbox_R = Rotation.from_rotvec(rotvec).as_matrix()
                bbox = o3d.geometry.OrientedBoundingBox(center, bbox_R, extent)

            human_dict = extract_human_points(pcd)
            human_points = human_dict["human_points"]
            np.save(f"{TRAIN_PATH}/{HUMANS_FOLDER}/{samp_name}.npy", human_points)
            inv_R_t = human_dict["inv_R_t"]
            np.save(f"{TRAIN_PATH}/{TRANSFORMATIONS_FOLDER}/{samp_name}.npy", inv_R_t)

            z_extent_offset = extent[2:3] - human_dict["human_height"]
            new_center = inv_R_t[:3, :3].T @ center + human_dict["t"]
            center_offset = new_center[:2] - np.median(human_points, axis=0)[:2]
            heading = Rotation.from_matrix(bbox_R).as_euler("ZXY")[0]
            hx_hy = np.array([np.cos(heading), np.sign(np.sin(heading))])
            label = np.concatenate([center_offset, extent[:2], z_extent_offset, hx_hy])
            np.save(f"{TRAIN_PATH}/{LABELS_FOLDER}/{samp_name}.npy", label)

            if visualize:
                bbox.color = BBOX_COLOR
                o3d.visualization.draw_geometries([pcd, bbox])


def get_train_samples(train_samples, bucket, visualize):
    for sample_dict in train_samples:
        get_train_sample(sample_dict, bucket, visualize)


def get_train_samples_parallel(d, bucket, visualize):
    all_train_samples = []
    for sample_dict in d["dataset"]["samples"]:
        label_status = sample_dict["labels"]["ground-truth"]["label_status"]
        if label_status == "LABELED":
            all_train_samples.append(sample_dict)

    n_jobs = 1 if visualize else multiprocessing.cpu_count()
    samples_per_job = int(np.ceil(len(all_train_samples) / n_jobs))
    procs = []
    for job in range(n_jobs):
        start = job * samples_per_job
        end = start + samples_per_job
        train_samples = all_train_samples[start:end]
        proc = multiprocessing.Process(
            target=get_train_samples, args=(train_samples, bucket, visualize)
        )
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


def get_test_sample(sample_dict, bucket, visualize):
    frames = sample_dict["attributes"]["frames"]
    for frame in frames:
        name = frame["name"]
        if os.path.exists(f"{TEST_PATH}/{HUMANS_FOLDER}/{name}.npy"):
            continue

        pcd_path = f"{DATASETS_PATH}/{PCDS_FOLDER}/{name}.pcd"
        if not os.path.exists(pcd_path):
            bucket.download_file(f"{S3_FOLDER}/{name}.pcd", pcd_path)

        pcd = o3d.io.read_point_cloud(pcd_path)
        human_dict = extract_human_points(pcd)
        if not human_dict:
            continue

        (human_points, inv_R_t) = (human_dict["human_points"], human_dict["inv_R_t"])
        np.save(f"{TEST_PATH}/{HUMANS_FOLDER}/{name}.npy", human_points)
        np.save(f"{TEST_PATH}/{TRANSFORMATIONS_FOLDER}/{name}.npy", inv_R_t)
        np.save(f"{TEST_PATH}/{HEIGHTS_FOLDER}/{name}.npy", human_dict["human_height"])

        if visualize:
            o3d.visualization.draw_geometries([pcd])


def get_test_samples(test_samples, bucket, visualize):
    for sample_dict in test_samples:
        get_test_sample(sample_dict, bucket, visualize)


def get_test_samples_parallel(d, bucket, visualize):
    all_test_samples = []
    for sample_dict in d["dataset"]["samples"]:
        label_status = sample_dict["labels"]["ground-truth"]["label_status"]
        if label_status == "PRELABELED":
            all_test_samples.append(sample_dict)

    n_jobs = 1 if visualize else multiprocessing.cpu_count()
    samples_per_job = int(np.ceil(len(all_test_samples) / n_jobs))
    procs = []
    for job in range(n_jobs):
        start = job * samples_per_job
        end = start + samples_per_job
        test_samples = all_test_samples[start:end]
        proc = multiprocessing.Process(
            target=get_test_samples, args=(test_samples, bucket, visualize)
        )
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


def main():
    folders = [HUMANS_FOLDER, LABELS_FOLDER, TRANSFORMATIONS_FOLDER, HEIGHTS_FOLDER]
    for folder in folders:
        os.makedirs(f"{TRAIN_PATH}/{folder}", exist_ok=True)
        os.makedirs(f"{TEST_PATH}/{folder}", exist_ok=True)

    s3 = boto3.resource("s3")
    bucket = s3.Bucket(S3_BUCKET)

    with open(LABELS_JSON) as f:
        d = json.load(f)

    try:
        visualize = sys.argv[1]
        assert visualize == "visualize"
    except IndexError:
        visualize = False

    get_train_samples_parallel(d, bucket, visualize)
    get_test_samples_parallel(d, bucket, visualize)


if __name__ == "__main__":
    main()
