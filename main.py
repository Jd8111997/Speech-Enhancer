
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from scipy.io import wavfile
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import Generator, Discriminator
from Generate_dataset import AudioDataset_Generator
from emphasis import emph 

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Train Audio Enhancement')
	parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
	parser.add_argument('--num_epochs', default=12, type=int, help='train epochs number')
	opt = parser.parse_args()
	run_time = time.strftime('%Y%m%d_%H%M', time.gmtime())
	sample_rate = 8000

	batch_size = opt.batch_size
	num_epochs = opt.num_epochs
	d_lr = 0.0001 
	g_lr = 0.0001
	gen_data_dir = 'Gen_results'
	out_path_root = 'segan_data_out'
	checkpoint_dir = 'model_weights'
	out_path = os.path.join(os.getcwd(), out_path_root, run_time)
	gen_data_path = os.path.join(out_path, gen_data_dir)
	if not os.path.exists(gen_data_path):
		os.makedirs(gen_data_path)
	checkpoint_path = os.path.join(out_path, checkpoint_dir)
	if not os.path.exists(checkpoint_path):
		os.makedirs(checkpoint_path)


	#Generating dataset

	print('Generating Dataset')
	train_dataset = AudioDataset_Generator(data_type = 'train')
	test_dataset = AudioDataset_Generator(data_type = 'test')
	train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
	test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=4)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	ref_batch = train_dataset.reference_batch(batch_size).to(device)
	discriminator = Discriminator().to(device)
	generator = Generator().to(device)
	ref_batch = Variable(ref_batch)

	print('Discriminator and Generator created')

	g_optimizer = optim.Adam(generator.parameters(), lr=g_lr)
	d_optimizer = optim.Adam(discriminator.parameters(), lr=d_lr)


	print('Training...')
	for epoch in range(num_epochs):
		train_bar = tqdm(train_data_loader)
		for train_batch, train_clean, train_noisy in train_bar:

			z = nn.init.normal_(torch.Tensor(train_batch.size(0), 512, 8)).to(device)
			train_batch, train_clean, train_noisy = train_batch.to(device), train_clean.to(device), train_noisy.to(device)
			z = Variable(z)
			train_batch, train_clean, train_noisy = Variable(train_batch), Variable(train_clean), Variable(train_noisy)

			# Training discriminator to recognize clean audio
			discriminator.zero_grad()
			output = discriminator(train_batch, ref_batch)
			clean_loss = torch.mean((output - 1.0) ** 2)
			clean_loss.backward()

			#Training discriminator to recognize generator's output as noisy
			generator.zero_grad()
			generated_output = generator(train_noisy, z)
			outputs = discriminator(torch.cat((generated_output, train_noisy), dim = 1), ref_batch)
			noisy_loss = torch.mean(outputs ** 2)
			noisy_loss.backward()

			d_optimizer.step()

			#Training Generator to so that discriminator recognized G(Z) as real

			generator.zero_grad()
			generated_output = generator(train_noisy, z)
			gen_noise_pair = torch.cat((generated_output, train_noisy), dim = 1)
			outputs = discriminator(gen_noise_pair, ref_batch)
			g_loss_ = 0.5 * torch.mean((outputs - 1.0) ** 2)
			l1_dist = torch.abs(torch.add(generated_output, torch.neg(train_clean)))
			g_cond_loss = 100 * torch.mean(l1_dist)
			g_loss = g_loss_ + g_cond_loss

			g_loss.backward()
			g_optimizer.step()

			train_bar.set_description('Epoch {}: d_clean_loss {:.4f}, d_noisy_loss {:.4f}, g_loss {:.4f}, g_conditional_loss {:.4f}'
                    .format(epoch + 1, clean_loss.item(), noisy_loss.item(), g_loss.item(), g_cond_loss.item()))

		#Testing models and generating samples
		if epoch % 4 == 0:


			test_bar = tqdm(test_data_loader, desc='Testing model and generating audios')
			for test_file_name, test_noisy in test_bar:
				z = nn.init.normal_(torch.Tensor(test_noisy.size(0), 512, 8))
				test_noisy, z = Variable(test_noisy.to(device)), Variable(z.to(device))
				fake_audio = generator(test_noisy, z).data.cpu().numpy()
				fake_audio = emph(fake_audio, emph_coeff=0.95, pre=False)

				for idx in range(fake_audio.shape[0]):
					generated_sample = fake_audio[idx]
					file_name = os.path.join(gen_data_path, '{}_e{}.wav'.format(test_file_name[idx].replace('.npy', ''), epoch + 1))
					wavfile.write(file_name, sample_rate, generated_sample.T)

			state_path = os.path.join(checkpoint_path, 'state-{}.pkl'.format(epoch + 1))
			state = {
				'discriminator' : discriminator.state_dict(),
				'generator': generator.state_dict(),
			}
			torch.save(state, state_path)









