# Speech-Enhancer
Speech Enhancing module based on paper [SEGAN](https://arxiv.org/abs/1703.09452)

## Requirements
- python v3.5.2 or higher
- librosa
- SoX
- pytorch v0.4.0

## Dataset
I have used a toy noisy speech dataset from the university of edinburgh [dataset](https://datashare.is.ed.ac.uk/handle/10283/1942).     
Download the train datasets and test datasets, then extract them into your directory and set the path of root_dir accordingly in `preprocess_final` module.


You can use other dataset, but change the path accordingly in `preprocess_final`.

## Pre-processing
You can change other parameter such as sample_rate, window_size etc in `preprocess_final`.    
Default sample rate is set to 8KHz and window_size is 8192. You can downsample the audio as your needs, by changing sample_rate in `downsample_8k`.   
By default the preprocessed datasets location is set to your current working directory.

## Training

```python
python main.py ----batch_size 128 --num_epochs 86
optional arguments:
--batch_size             train batch size [default value is 128]
--num_epochs             train epochs number [default value is 12]
```
At every four epoch the test results and model weights will be saved in `segan_data_out`.    
Again adjust the paths and parameters in `main.py` according to your needs.  

## Enhancing audio

```python
python Generate_audio.py ----file_name p212_982.wav --time_stamp 20200816_0531 --state state-5.pkl
```
Give a noisy audio clip as an input to `Generate_audio` and set the appropriate time_stame and state from segan_data_out that is created from `main.py` to load saved weights of generator.
