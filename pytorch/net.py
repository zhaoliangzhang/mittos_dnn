import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import os
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.onnx as torch_onnx

class Mynet(nn.Module):
	def __init__(self):
		super(Mynet, self).__init__()
		self.fc1 = nn.Linear(40, 128, bias=False)
		self.fc2 = nn.Linear(128, 1, bias=False)

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)

		return x

def train(epoch, dataloader, module):
	for i in range(epoch):
		for batch_idx, (data, target) in enumerate(dataloader):
			data, target = Variable(data), Variable(target)
			optimizer = optim.SGD(module.parameters(), lr=0.01, momentum=0.9)
			optimizer.zero_grad()
			output = module(data)
			loss = F.kl_div(output, target)
			loss.backward()
			optimizer.step()
			if batch_idx % 200 == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(i, batch_idx * len(data), len(dataloader.dataset), 100. * batch_idx / len(dataloader), loss.data))
	torch.save(module, "./module/my_model.pkl")
	dummy_input = torch.randn(40)
	torch_onnx.export(module, args=dummy_input, f="./module/my_model.onnx")

def test(dataloader, module):
	test_loss = 0
	correct = 0
	print("start test")
	for data, target in dataloader:
		data, target = Variable(data), Variable(target)
		torch.no_grad()
		output = module(data)
		test_loss += F.kl_div(output.float(), target, size_average=False).data
		predict = output.data.float()
		correct += torch.eq(target,predict).sum()
	test_loss /= len(dataloader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {})\n'.format(test_loss, float(correct.sum())/float(len(dataloader.dataset))))