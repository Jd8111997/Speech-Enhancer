import os
import numpy as np 
import torch
import torch.nn as nn
from scipy.io import wavfile
from torch.autograd import Variable
import argparse
from tqdm import tqdm
from model import Generator
from emphasis import emph
from preprocess_final import slice_signal, window_size, sample_rate

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Testing Single Audio Enhancement')
	parser.add_argument('--file_name', type=str, required=True, help='audio file name')
	parser.add_argument('--time_stamp', type=str, required=True, help='Time-stamp')
	parser.add_argument('--state', type=str, required=True, help='state')
	opt = parser.parse_args()
	File_name = opt.file_name
	time_stamp_no = opt.time_stamp
	state = opt.state
	generator = Generator()
	generator.load_state_dict(torch.load('segan_data_out/' + time_stamp_no + '/model_weights/' + state, map_location='cpu'))
	if torch.cuda.is_available():
		generator.cuda()
	noisy_slices = slice_signal(File_name, window_size, 1, sample_rate)
	enhanced_audio = []
	for noisy_slice in tqdm(noisy_slices, desc = 'Generating enhanced audio'):
		z = nn.init.normal_(torch.Tensor(1, 512, 8))
		noisy_slice = torch.from_numpy(emphasis(noisy_slice[np.newaxis, np.newaxis, :])).type(torch.FloatTensor)
		if torch.cuda.is_available():
			noisy_slice, z = noisy_slice.cuda(), z.cuda()
		noisy_slice,z = Variable(noisy_slice), Variable(z)
		generated_audio = generator(noisy_slice, z).data.cpu().numpy()
		generated_audio = emphasis(generated_speech, emph_coeff=0.95, pre=False)
		generated_audio = generated_speech.reshape(-1)
		enhanced_audio.append(generated_audio)

	enhanced_audio = np.array(enhanced_audio).reshape(1, -1)
	file_name = os.path.join(os.path.dirname(File_name), 'enhanced_{}.wav'.format(os.path.basename(File_name).split('.')[0]))
	wavfile.write(file_name, sample_rate, enhanced_audio.T)
