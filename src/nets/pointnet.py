import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Conv1d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size)
        self.bn = nn.BatchNorm1d(output_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class Linear(nn.Module):
    def __init__(self, input_features, output_features):
        super(Linear, self).__init__()
        self.fc = nn.Linear(input_features, output_features)
        self.bn = nn.BatchNorm1d(output_features)

    def forward(self, x):
        return F.relu(self.bn(self.fc(x)))

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.k = k

        self.conv1 = Conv1d(k, 64, 1)
        self.conv2 = Conv1d(64, 128, 1)
        self.conv3 = Conv1d(128, 1024, 1)

        self.fc1 = Linear(1024, 512)
        self.fc2 = Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = torch.max(out, 2, keepdim=True)[0]
        out = out.view(-1, 1024)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        iden = torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).view(1, self.k * self.k).repeat(x.size()[0], 1)

        if x.is_cuda:
            iden = iden.cuda()

        out += iden
        out = out.view(-1, self.k, self.k)

        return out

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.global_feat = global_feat
        self.feature_transform = feature_transform

        self.stn = STNkd(k=3)
        if self.feature_transform:
            self.fstn = STNkd(k=64)

        self.conv1 = Conv1d(3, 64, 1)
        self.conv2 = Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn = nn.BatchNorm1d(1024)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans)
        x = x.transpose(2,1)
        out = self.conv1(x)

        if self.feature_transform:
            trans_feat = self.fstn(out)
            out = out.transpose(2,1)
            out = torch.bmm(out, trans_feat)
            out = out.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = out
        out = self.conv2(out)
        out = self.bn(self.conv3(out))
        out = torch.max(out, 2, keepdim=True)[0]
        out = out.view(-1, 1024)
        if self.global_feat:
            return out, trans, trans_feat
        else:
            out = out.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([out, pointfeat], 1), trans, trans_feat

class PointNetSegmentation(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetSegmentation, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = Conv1d(1088, 512, 1)
        self.conv2 = Conv1d(512, 256, 1)
        self.conv3 = Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.k, 1)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        out, trans, trans_feat = self.feat(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.transpose(2,1).contiguous()
        out = F.log_softmax(out.view(-1, self.k), dim=-1)
        out = out.view(batchsize, n_pts, self.k)
        return out, trans, trans_feat

def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(
        torch.norm(torch.bmm(trans, trans.transpose(2,1) - I), dim=(1,2)))
    return loss

if __name__ == '__main__':
    sim_data = torch.rand(32,3,2500)
    trans = STNkd(k=3)
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_reguliarzer(out))

    sim_data_64d = torch.rand(32, 64, 2500)
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_reguliarzer(out))

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    seg = PointNetSegmentation(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())