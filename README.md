# Model-Assisted Parking Lot Labeling

This repository contains code that was used to train a PointNet++ bounding box regression model on data that was collected using the parking lot protocol of [Paved2Paradise](https://github.com/airalcorn2/paved2paradise).
The code does not run as is.
You will need to collect your own parking lot data and modify the [`prepare_datasets.py`](prepare_datasets.py) and [`upload_segments_labels.py`](upload_segments_labels.py) scripts to reflect the idiosyncrasies of your datasets.
The model can be trained on a small number of labeled sequences (e.g., one), and then used to predict bounding boxes for the remaining samples, which can then be refined in Segments.ai.

## Installing the necessary Python packages

```bash
pip3 install -r requirements.txt
```

## Generating the training/test data

```bash
python3 prepare_datasets.py
```

## Training the PointNet++ model

Training the model requires PyTorch3D, which is easiest used in a container, so start the container:

```bash
# The path to this repo.
REPO_PATH=/home/michael/Projects/parking-lot-pointnetplusplus
# Should be the same as the DATASETS_PATH variable in env_vars.py.
DATASETS_PATH=/media/michael/Extra/aa_field_day
docker run -v ${REPO_PATH}:/workspace/pointnetplusplus -v ${DATASETS_PATH}:/workspace/datasets --runtime nvidia -it pytorch3d
```

After you start the container, log in to Weights & Biases:

```bash
wandb login --host <your_wandb_url>
```

and then train the model:

```bash
cd pointnetplusplus
python3 train.py
```

## Generating the bounding box predictions

```bash
python3 predict_bboxes.py <run_id>
```

where `<run_id>` is the ID associated with the Weights & Biases run, e.g.:

```bash
python3 predict_bboxes.py 2gyfwhhj
```

To upload the bounding box predictions to Segments.ai, run:

```bash
python3 upload_segments_labels.py [visualize]
```

where `visualize` is an optional argument that when included will visualize the bounding box predictions.
