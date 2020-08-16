import os

import numpy as np
import torch
from torch.utils import data

def emph(signal_batch, emph_coeff=0.95, pre=True):

	"""
	Pre-emphasis or De-emphasis of higher frequencies given a batch of signal.

	"""

	result = np.zeros(signal_batch.shape)
	for sample_idx, sample in enumerate(signal_batch):

		for ch, channel_data in enumerate(sample):

			if pre:
				result[sample_idx][ch] = np.append(channel_data[0], channel_data[1:] - emph_coeff * channel_data[:-1])
			else:
				result[sample_idx][ch] = np.append(channel_data[0], channel_data[1:] + emph_coeff * channel_data[:-1])
	return result
	