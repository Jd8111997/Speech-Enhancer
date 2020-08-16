import torch
from torch.utils import data
import numpy as np 
import os
from emphasis import emph 
from preprocess_final import serialized_train_folder, serialized_test_folder

 
class AudioDataset_Generator(data.Dataset):

	def __init__(self, data_type):

		if data_type == 'train':
			data_path = serialized_train_folder
		else:
			data_path = serialized_test_folder
		if not os.path.exists(data_path):
			raise FileNotFoundError('The {} data folder does not exist!'.format(data_type))
		self.data_type = data_type
		self.file_names = [os.path.join(data_path, filename) for filename in os.listdir(data_path)]

	def reference_batch(self, batch_size):

		ref_file_names = np.random.choice(self.file_names, batch_size)
		ref_batch = torch.from_numpy(np.stack([np.load(f) for f in ref_file_names]))
		ref_batch = emph(ref_batch, emph_coeff = 0.95)
		return torch.from_numpy(ref_batch).type(torch.FloatTensor)

	def __getitem__(self, idx):

		pair = np.load(self.file_names[idx])
		pair = emph(pair[np.newaxis, :, :], emph_coeff=0.95).reshape(2, -1)
		noisy = pair[1].reshape(1, -1)
		if self.data_type == 'train':
			clean = pair[0].reshape(1, -1)
			return torch.from_numpy(pair).type(torch.FloatTensor), torch.from_numpy(clean).type(torch.FloatTensor), torch.from_numpy(noisy).type(torch.FloatTensor)
		else:
			return os.path.basename(self.file_names[idx]), torch.from_numpy(noisy).type(torch.FloatTensor)

	def __len__(self):

		return len(self.file_names)

