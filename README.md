# FAST-TO-WAV - From fast third-octave data to high quality audio waveform

This repository provides code to transform fast third-octave data into high-quality audio waveforms. The model is based on the diffusion architecture described in the paper:  
**"Diffusion-based Spectral Super-Resolution of Third Octave Acoustic Sensor Data: Is Privacy at Risk?"**  
It is trained on the **TAU Urban Acoustic Scenes 2020 Mobile, Development dataset**.

## Setup

First, install the required dependencies using the following command in a new Python 3.10.12 environment:
```
pip install -r requirements.txt
```

## Test fto-to-wav with your own audio files

You can evaluate the waveform reconstruction quality by converting an audio file of your choice into fast third-octave data and then reconstructing it using the `fast-to-wav` algorithm. This allows you to compare the original and reconstructed versions by ear.

1. Place your audio file (e.g., `XXX.wav`) into the `./audio/` directory.  
2. Run the following command:

```
python3 wav_to_fast_to_wav.py XXX.wav
```

The computation typically takes around **4× the duration** of the original signal.  
For example, processing a 1-minute fast third-octave file will take approximately 4 minutes.

## Run super-resolution with your own fast third-octave dataset

## Run Super-Resolution on Your Own Fast Third-Octave Dataset

To apply `fast-to-wav` on your own third-octave data, place your fast third-octave data as a `.csv` file inside the `spectral_data/` directory. The data must cover frequency bands from **125 Hz to 10 kHz**, i.e., 20 bands total. The CSV file must include the following column names (in order):

```
epoch,Fast_125,Fast_160,Fast_200,Fast_250,Fast_315,Fast_400,Fast_500,Fast_630,Fast_800,Fast_1000,Fast_1250,Fast_1600,Fast_2000,Fast_2500,Fast_3150,Fast_4000,Fast_5000,Fast_6300,Fast_8000,Fast_10000
```

> ⚠️ **Note:** The `epoch` column is ignored during computation, so its format can be arbitrary.

To generate the waveform on the csv file named XXX.csv, run:

```
python3 fast_to_wav.py -n XXX.csv
```

The output waveform 
file is stored in directory `./predictions/. The computation typically takes around **4× the duration** of the original signal. 
For example, processing a 1-minute fast third-octave file will take approximately 4 minutes.

By default, the algorithm applies a **–88 dB** offset to convert the data into the dBFS format.  
You can manually set the offset to `x` dB by running:

```
python3 fast_to_wav.py -n XXX.csv -dbo x
```

To automatically determine an appropriate dB offset based on your data, use:

```
python3 fast_to_wav.py -n XXX.csv -gdbo
```

The terminal will display a suggested offset value that you can reuse with the `-dbo` option.

## Companion page

Please check the companion page for audio examples:
https://modantailleur.github.io/fast-to-wav/