import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class STNKd(nn.Module):
    # T-Net a.k.a. Spatial Transformer Network
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        """
        Input: [B,N,k]
        Output: [B,k,k]
        """
        B = x.shape[0]
        device = x.device

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]

        x = self.fc(x)
        
        # Followed the original implementation to initialize a matrix as I.
        identity = (
            Variable(torch.eye(self.k, dtype=torch.float))
            .reshape(1, self.k * self.k)
            .expand(B, -1)
            .to(device)
        )
        x = x + identity
        x = x.reshape(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    """
    def __init__(
        self,
        input_transform: bool = False,
        feature_transform: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.stn3 = STNKd(k=3)
        if self.feature_transform:
            self.stn64 = STNKd(k=64)

        # point-wise mlp
        # TODO : Implement point-wise mlp model based on PointNet Architecture.
        self.conv1 = nn.Sequential(nn.Conv1d(3,64,1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64,128,1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128,1024,1), nn.BatchNorm1d(1024))

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - Global feature: [B,1024]
            - ...
        """

        # TODO : Implement forward function.
        x = pointcloud  # B,N,3
        B, N, _ = x.shape
        device = x.device

        x = x.to('cuda')

        if self.input_transform:
            x = x.transpose(2,1)
            i_trans = self.stn3(x) # B,3,3

            # matmul x * i_trans
            x = x.transpose(2,1)
            x = torch.bmm(x, i_trans) # B,N,3

        x = x.transpose(2,1)
        x = F.relu(self.conv1(x))    # 64

        feat_trans = None
        f_trans = None
        if self.feature_transform:
            f_trans = self.stn64(x)
            # matmul x * f_trans [n, 3, 3]
            x = torch.bmm(x.transpose(2,1), f_trans)
            x = x.transpose(2,1)
            feat_trans = x

        x = F.relu(self.conv2(x))    # 128
        x = self.conv3(x)    # 1024

        # Max-pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.reshape(-1, 1024)
        return x, f_trans, feat_trans


class PointNetCls(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes
        
        # extracts max-pooled features
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)
        
        # returns the final logits from the max-pooled features.
        # TODO : Implement MLP that takes global feature as an input and return logits.
        
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - logits [B,num_classes]
            - ...
        """
        # TODO : Implement forward function.
        global_feature, f_trans, feat_trans = self.pointnet_feat(pointcloud)
        x = self.fc(global_feature)
        return F.log_softmax(x, dim=-1), feat_trans


class PointNetPartSeg(nn.Module):
    def __init__(self, num_counts = 50, input_transform = False, feature_transform = False):
        super().__init__()

        # returns the logits for m part labels each point (m = # of parts = 50).
        # TODO: Implement part segmentation model based on PointNet Architecture.

        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)

        self.num_counts = num_counts
        self.conv1 = nn.Sequential(nn.Conv1d(1088,512,1), nn.BatchNorm1d(512))
        self.conv2 = nn.Sequential(nn.Conv1d(512,256,1), nn.BatchNorm1d(256))
        self.conv3 = nn.Sequential(nn.Conv1d(256,128,1), nn.BatchNorm1d(128))
        self.conv4 = nn.Conv1d(128, self.num_counts, 1)


    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - logits: [B,50,N] | 50: # of point labels
            - ...
        """
        # TODO: Implement forward function.
        B, N, _ = pointcloud.size()
        global_feature, f_trans, feat_trans = self.pointnet_feat(pointcloud)
        global_feature = global_feature.view(-1, 1024, 1).repeat(1, 1, N)
        x = torch.cat([global_feature, feat_trans], 1) # B, 1088, N

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x) # 32,50,2048

        x = F.log_softmax(x, dim = -1) # 32,50,2048
        
        return x.view(B, self.num_counts, N), f_trans
        

class PointNetAutoEncoder(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.pointnet_feat = PointNetFeat()

        # Decoder is just a simple MLP that outputs N x 3 (x,y,z) coordinates.
        n1 = int(num_points/4)
        n2 = int(num_points/2)
        self.fc = nn.Sequential(
            nn.Linear(1024, n1),
            nn.BatchNorm1d(n1),
            nn.ReLU(),
            nn.Linear(n1, n2),
            nn.BatchNorm1d(n2),
            nn.ReLU(),
            nn.Linear(n2, num_points),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(num_points),
            nn.ReLU(),
            nn.Linear(num_points, num_points*3),
        )


    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - pointcloud [B,N,3]
            - ...
        """
        # TODO : Implement forward function.
        B, N, _ = pointcloud.shape
        global_feature, _, _ = self.pointnet_feat(pointcloud)
        x = self.fc(global_feature)
        return x.view(B, N, 3)


def get_orthogonal_loss(feat_trans, reg_weight=1e-3):
    """
    a regularization loss that enforces a transformation matrix to be a rotation matrix.
    Property of rotation matrix A: A*A^T = I
    """
    if feat_trans is None:
        return 0

    B, K = feat_trans.shape[:2]
    device = feat_trans.device

    identity = torch.eye(K).to(device)[None].expand(B, -1, -1)
    mat_square = torch.bmm(feat_trans, feat_trans.transpose(1, 2))

    mat_diff = (identity - mat_square).reshape(B, -1)

    return reg_weight * mat_diff.norm(dim=1).mean()
