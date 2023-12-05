config = {
    "model": "pointnet++",
    "train_p": 0.975,
    "num_workers": 4,
    "batch_size": 32,
    "model_args": {
        "position_encodings": 0,
        "sa_layers": [
            {
                "sample_points": 1024,
                "max_neighbors": 32,
                "radius": 0.1,
                "mlp_layers": [32, 32, 64],
            },
            {
                "sample_points": 256,
                "max_neighbors": 32,
                "radius": 0.2,
                "mlp_layers": [64, 64, 128],
            },
            {
                "sample_points": 64,
                "max_neighbors": 32,
                "radius": 0.4,
                "mlp_layers": [128, 128, 256],
            },
            {
                "sample_points": 16,
                "max_neighbors": 32,
                "radius": 0.8,
                "mlp_layers": [256, 256, 512],
            },
        ],
        "last_feats": 512,
    },
    "lr": 1e-3,
    "patience": 10,
    "max_lr_drops": 2,
    "lr_reducer": 0.5,
    "epochs": 650,
}
