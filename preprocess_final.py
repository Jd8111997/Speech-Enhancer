import os
import subprocess
import librosa
import numpy as np
from tqdm import tqdm

root_dir = '../'
clean_train_folder = 'clean_trainset'
noisy_train_folder = 'noisy_trainset'
clean_test_folder = 'clean_testset'
noisy_test_folder = 'noisy_testset'
downsampled_clean_train_folder = 'downsampled_clean_train'
downsampled_clean_test_folder = 'downsampled_clean_test'
downsampled_noisy_train_folder = 'downsampled_noisy_train'
downsampled_noisy_test_folder = 'downsampled_noisy_test'
serialized_train_folder = 'serialized_train_data'
serialized_test_folder = 'serialized_test_data'
window_size = 2 ** 13  # about 1 second of samples
sample_rate = 8000



def slice_signal(file, window_size, stride, sample_rate):
    """
    Helper function for slicing the audio file
    by window size and sample rate with [1-stride] percent overlap (default 50%).
    """
    wav, sr = librosa.load(file, sr=sample_rate)
    hop = int(window_size * stride)
    slices = []
    for end_idx in range(window_size, len(wav), hop):
        start_idx = end_idx - window_size
        slice_sig = wav[start_idx:end_idx]
        slices.append(slice_sig)
    return slices


def process_and_serialize(data_type):
    """
    Serialize, down-sample the sliced signals and save on separate folder.
    """
    stride = 0.5

    if data_type == 'train':
        clean_folder = downsampled_clean_train_folder
        noisy_folder = downsampled_noisy_train_folder
        serialized_folder = serialized_train_folder
    else:
        clean_folder = downsampled_clean_test_folder
        noisy_folder = downsampled_noisy_test_folder
        serialized_folder = serialized_test_folder
    serialized_folder = os.path.join(root_dir, serialized_folder)
    if not os.path.exists(serialized_folder):
        os.makedirs(serialized_folder)

    # walk through the path, slice the audio file, and save the serialized result
    clean_data_path = os.path.join(root_dir, clean_folder)
    noisy_data_path = os.path.join(root_dir, noisy_folder)
    for root, dirs, files in os.walk(clean_data_path):
        if len(files) == 0:
            continue
        for filename in tqdm(files, desc='Serialize and down-sample {} audios'.format(data_type)):
            clean_file = os.path.join(clean_data_path, filename)
            noisy_file = os.path.join(noisy_data_path, filename)
            # slice both clean signal and noisy signal
            clean_sliced = slice_signal(clean_file, window_size, stride, sample_rate)
            noisy_sliced = slice_signal(noisy_file, window_size, stride, sample_rate)
            # serialize - file format goes [original_file]_[slice_number].npy
            # ex) p293_154.wav_5.npy denotes 5th slice of p293_154.wav file
            for idx, slice_tuple in enumerate(zip(clean_sliced, noisy_sliced)):
                pair = np.array([slice_tuple[0], slice_tuple[1]])
                np.save(os.path.join(serialized_folder, '{}_{}'.format(filename, idx)), arr=pair)


def data_verify(data_type):
    """
    Verifies the length of each data after pre-process.
    """
    if data_type == 'train':
        serialized_folder = serialized_train_folder
    else:
        serialized_folder = serialized_test_folder

    serialize_path = os.path.join(root_dir, serialized_folder)
    for root, dirs, files in os.walk(serialize_path):
        for filename in tqdm(files, desc='Verify serialized {} audios'.format(data_type)):
            data_pair = np.load(os.path.join(root, filename))
            if data_pair.shape[1] != window_size:
                print('Snippet length not {} : {} instead'.format(window_size, data_pair.shape[1]))
                break

def downsample_8k(root_dir, input_dir, output_dir):

    down_output_dir = os.path.join(root_dir, output_dir)
    if not os.path.exists(down_output_dir):
        os.makedirs(down_output_dir)

    for dirname, dirs, files in os.walk(os.path.join(root_dir, input_dir)):
        for filename in files:
            input_file_path = os.path.abspath(os.path.join(dirname, filename))
            out_file_path = os.path.join(down_output_dir, filename)
            print('Downsampling : {}'.format(input_file_path))
            subprocess.run(
                'sox {} -r 8k {}'
                .format(input_file_path, out_file_path),
                shell = True, check=True)





if __name__ == '__main__':
    downsample_8k(root_dir, clean_train_folder, downsampled_clean_train_folder)
    downsample_8k(root_dir, noisy_train_folder, downsampled_noisy_train_folder)
    downsample_8k(root_dir, clean_test_folder, downsampled_clean_test_folder)
    downsample_8k(root_dir, noisy_test_folder, downsampled_noisy_test_folder)
    process_and_serialize('train')
    data_verify('train')
    process_and_serialize('test')
    data_verify('test')
