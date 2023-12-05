LABELS_JSON = "AAFieldDay.json"
S3_BUCKET = "your-s3-bucket"
S3_FOLDER = "pcds"
DATASETS_PATH = "/media/michael/Extra/aa_field_day"
TRAIN_PATH = f"{DATASETS_PATH}/train"
TEST_PATH = f"{DATASETS_PATH}/test"
PCDS_FOLDER = "pcds"
HUMANS_FOLDER = "humans"
LABELS_FOLDER = "labels"
TRANSFORMATIONS_FOLDER = "transformations"
HEIGHTS_FOLDER = "heights"
BBOX_COLOR = [1, 0, 0]
ACTIONS = ["Rotation", "Normal CW", "Normal CCW", "Crouch CW", "Crouch CCW"]
Y_THRESH = 2.0
DOT_DIST = 5.0
X_MIN = DOT_DIST - 2
X_MAX = DOT_DIST + 2
# The Ouster was 34"/0.8636 m off the ground. This adds a little fudge factor.
Z_THRESH = -0.7
BUFFER = 0.25

# Defines the grid area for leveling the ground plane.
MAX_X = 7.0
MAX_Y = 3.0
# The number of points along one side of the grid used to estimate the ground plane.
N_GRID_POINTS = 100

DOCKER_DATASETS_PATH = "/workspace/datasets"
DOCKER_TRAIN_PATH = f"{DOCKER_DATASETS_PATH}/train"
DOCKER_TEST_PATH = f"{DOCKER_DATASETS_PATH}/test"

DATA_DICT_F = "train_valid.json"
BEST_PARAMS_F = "best_params.pth"
WANDB_PROJECT = "parking_lot"
ENTITY = "airalcorn2"
MODEL_F = "pointnetplusplus.py"

WANDB_RUNS = "wandb_runs"
CONFIG_F = "config.pydict"
