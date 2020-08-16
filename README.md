# Speech-Enhancer
Speech Enhancing module based on paper [SEGAN](https://arxiv.org/abs/1703.09452)

## Requirements
- python v3.5.2 or higher
- librosa
- SoX
- pytorch v0.4.0

## Dataset
I have used a toy noisy speech dataset from the university of edinburgh [dataset](https://datashare.is.ed.ac.uk/handle/10283/1942).     
Download the train datasets and test datasets, then extract them into your directory and set the path of root_dir accordingly in preprocess_final module.


You can use other dataset, but change the path accordingly in preprocess_final.

## Pre-processing
You can change other parameter such as sample_rate, window_size etc. Default sample rate is set to 8KHz and window_size is 8192.
```python
preprocess_final.py
```
