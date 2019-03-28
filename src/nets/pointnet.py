import torch
import torch.nn as nn
import torch.nn.functional as F

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
		super(STN3d, self).__init__()
		self.k = k

		self.conv1 = Conv1d(4, 64, 1)
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
		self.conv3 = Conv1d(128, 1024, 1)

	def forward(self, x):
