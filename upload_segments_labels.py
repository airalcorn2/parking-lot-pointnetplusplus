# See: https://docs.segments.ai/reference/label-types#cuboid-annotation,
# and: https://docs.segments.ai/reference/label-types#cuboid-label-1.

import json
import numpy as np
import open3d as o3d
import os
import sys

from env_vars import *
from scipy.spatial.transform import Rotation
from segments import SegmentsClient


def main():
    with open("segments_api_key.txt") as f:
        api_key = f.read().strip()

    segments_client = SegmentsClient(api_key)

    with open(LABELS_JSON) as f:
        d = json.load(f)

    try:
        visualize = sys.argv[1]
        assert visualize == "visualize"
    except IndexError:
        visualize = False

    for sample_dict in d["dataset"]["samples"]:
        name = sample_dict["name"]
        if sample_dict["labels"]["ground-truth"]["label_status"] == "LABELED":
            continue

        print(name)
        sample_uuid = sample_dict["uuid"]
        frames = []
        for idx, frame in enumerate(sample_dict["attributes"]["frames"]):
            name = frame["name"]
            label_f = f"{name}.npy"
            label_path = f"{TEST_PATH}/{LABELS_FOLDER}/{label_f}"
            annotations = []
            if os.path.exists(label_path):
                labels = np.load(label_path)
                center = labels[:3]
                extent = labels[3:6]
                bbox_R = labels[6:].reshape(3, 3)

                r = Rotation.from_matrix(bbox_R)
                yaw = r.as_euler("ZXY")[0]
                q = r.as_quat()
                bbox_annotation = {
                    "id": 1,
                    "category_id": 1,
                    "type": "cuboid",
                    # See: https://stackoverflow.com/questions/64154850/convert-dictionary-to-a-json-in-python.
                    "position": {
                        "x": float(center[0]),
                        "y": float(center[1]),
                        "z": float(center[2]),
                    },
                    "dimensions": {
                        "x": float(extent[0]),
                        "y": float(extent[1]),
                        "z": float(extent[2]),
                    },
                    "yaw": float(yaw),
                    # Only when 3D cuboid rotation is enabled in dataset settings.
                    "rotation": {
                        "qx": float(q[0]),
                        "qy": float(q[1]),
                        "qz": float(q[2]),
                        "qw": float(q[3]),
                    },
                    "track_id": 1,
                    "is_keyframe": True,
                    "index": idx,
                }
                annotations = [bbox_annotation]

                if visualize:
                    print(center)
                    print(extent)
                    pcd_f = f"{name}.pcd"
                    pcd = o3d.io.read_point_cloud(
                        f"{DATASETS_PATH}/{PCDS_FOLDER}/{pcd_f}"
                    )
                    bbox = o3d.geometry.OrientedBoundingBox(center, bbox_R, extent)
                    bbox.color = BBOX_COLOR
                    o3d.visualization.draw_geometries([pcd, bbox])

            frames.append({"format_version": "0.2", "annotations": annotations})

        attributes = {"format_version": "0.1", "frames": frames}
        if not visualize:
            segments_client.add_label(sample_uuid, "ground-truth", attributes)


if __name__ == "__main__":
    main()
