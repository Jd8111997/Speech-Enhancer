import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.modules import Module

class VirtualBatchNorm1d(Module):

	"""

	Module for Virtual Batch Normalizaton

	Implementation borrowed and modified from Rafael_Valle's code + help of SimonW from this discussion thread:
	https://discuss.pytorch.org/t/parameter-grad-of-conv-weight-is-none-after-virtual-batch-normalization/9036
	"""

	def __init__(self, num_features, eps=1e-5):
		super().__init__()
		self.num_features = num_features
		self.eps = eps
		self.gamma = Parameter(torch.normal(mean=1.0, std=0.02, size=(1, num_features, 1)))
		self.beta = Parameter(torch.zeros(1, num_features, 1))

	def get_stats(self, x):

		mean = x.mean(2, keepdim=True).mean(0, keepdim=True)
		mean_sq = (x ** 2).mean(2, keepdim=True).mean(0, keepdim=True)
		return mean, mean_sq

	def forward(self, x, ref_mean, ref_mean_sq):

		mean, mean_sq = self.get_stats(x)
		if ref_mean is None or ref_mean_sq is None:
			mean = mean.clone().detach()
			mean_sq = mean_sq.clone().detach()
			out = self.normalize(x, mean, mean_sq)
		else:
			batch_size = x.size(0)
			new_coeff = 1. / (batch_size + 1.)
			old_coeff = 1. - new_coeff
			mean = new_coeff * mean + old_coeff * ref_mean
			mean_sq = new_coeff * mean_sq + old_coeff * ref_mean_sq
			out = self.normalize(x, mean, mean_sq)
		return out, mean, mean_sq

	def normalize(self, x, mean, mean_sq):

		assert mean_sq is not None
		assert mean is not None
		assert len(x.size()) == 3
		if mean.size(1) != self.num_features:
			raise Exception('Mean tensor size not equal to number of features : given {}, expected {}'
                            .format(mean.size(1), self.num_features))
		if mean_sq.size(1) != self.num_features:
			raise Exception('Squared mean tensor size not equal to number of features : given {}, expected {}'
                            .format(mean_sq.size(1), self.num_features))

		std = torch.sqrt(self.eps + mean_sq - mean ** 2)
		x = x - mean
		x = x / std
		x = x * self.gamma
		x = x + self.beta
		return x

	def __repr__(self):
		return ('{name}(num_features={num_features}, eps={eps}'
                .format(name=self.__class__.__name__, **self.__dict__))
		
