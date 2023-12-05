import torch

from pytorch3d.ops import ball_query, sample_farthest_points
from torch import nn

# See: https://github.com/charlesq34/pointnet2/blob/42926632a3c33461aebfbee2d829098b30a23aaa/utils/pointnet_util.py#L212.
DIST_MIN = 1e-10


class SetAbstraction(nn.Module):
    def __init__(self, sample_points, max_neighbors, radius, in_feats, mlp_layers):
        super().__init__()
        self.sample_points = sample_points
        self.max_neighbors = max_neighbors
        self.radius = radius

        mlp = []
        in_feats = 3 + in_feats
        for out_feats in mlp_layers:
            mlp.append(nn.Conv2d(in_feats, out_feats, 1))
            mlp.append(nn.BatchNorm2d(out_feats))
            mlp.append(nn.ReLU())
            in_feats = out_feats

        self.mlp = nn.Sequential(*mlp)

    def forward(self, xyzs, feats):
        # See "Sampling layer." in Section 3.2.
        (center_xyzs, _) = sample_farthest_points(xyzs, K=self.sample_points)

        # See "Grouping layer." in Section 3.2.
        (_, group_idxs, group_xyzs) = ball_query(
            center_xyzs, xyzs, K=self.max_neighbors, radius=self.radius
        )

        # See "PointNet layer." in Section 3.2. The grouped points are translated
        # relative to their associated query points.
        group_xyzs = group_xyzs - center_xyzs.unsqueeze(2)
        if feats is None:
            group_feats = group_xyzs

        else:
            batch_idxs = torch.arange(len(xyzs))[:, None, None]
            group_feats = feats[batch_idxs, group_idxs]
            # See: https://github.com/charlesq34/pointnet2/blob/42926632a3c33461aebfbee2d829098b30a23aaa/utils/pointnet_util.py#L50.
            group_feats = torch.cat([group_xyzs, group_feats], dim=-1)
            # ball_query returns -1 when there are fewer than self.max_neighbors points
            # in a query point's neighborhood. As a result, those features are filled
            # with zeros.
            group_feats = group_feats * (group_idxs != -1).unsqueeze(3).float()

        # PointNet.
        group_feats = group_feats.permute(0, 3, 1, 2)
        group_feats = self.mlp(group_feats).permute(0, 2, 3, 1)
        (center_feats, _) = group_feats.max(dim=2)
        return (center_xyzs, center_feats)


class PointNetPlusPlus(nn.Module):
    def __init__(self, position_encodings, in_feats, sa_layers, last_feats, out_feats):
        super().__init__()
        self.L = position_encodings
        in_feats = (1 + 2 * position_encodings) * in_feats
        sas = []
        prev_in_feats = [in_feats]
        for sa_layer in sa_layers:
            sa_layer["in_feats"] = in_feats
            sas.append(SetAbstraction(**sa_layer))
            in_feats = sa_layer["mlp_layers"][-1]
            prev_in_feats.append(in_feats)

        self.last_radius = sas[-1].radius
        self.sas = nn.ModuleList(sas)

        self.final_layer = nn.Sequential(
            nn.Linear(in_feats, last_feats),
            nn.ReLU(),
            nn.Linear(last_feats, last_feats),
            nn.ReLU(),
            nn.Linear(last_feats, out_feats),
            nn.Tanh(),
        )

    def forward(self, xyzs, feats):
        feats_encoded = [feats]
        for l_pos in range(self.L):
            feats_encoded.append(torch.sin(2**l_pos * torch.pi * feats))
            feats_encoded.append(torch.cos(2**l_pos * torch.pi * feats))

        feats = torch.cat(feats_encoded, dim=2)

        skip_links = []
        for sa in self.sas:
            skip_links.append((xyzs, feats))
            (xyzs, feats) = sa(xyzs, feats)

        feats = feats.max(1)[0]
        preds = self.final_layer(feats)

        return preds
