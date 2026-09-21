# Deep Learning Based Real-Time Speech Enhancement and Noise Suppression

A deep learning-based speech enhancement system that reduces background noise from speech using spectrogram-based neural networks. The project implements traditional DSP, CNN, U-Net, and Mask U-Net approaches and provides both a real-time audio enhancement prototype and a web application.

---

## Project Overview

Speech recorded in real-world environments often contains unwanted background noise such as white noise, low-frequency noise, and mixed environmental noise.

This project addresses the problem by converting noisy speech into a time-frequency representation using Short-Time Fourier Transform (STFT), processing the spectrogram using deep learning, and reconstructing enhanced speech using inverse STFT (iSTFT).

The project progresses from a traditional DSP baseline to deep learning models and finally integrates the trained Mask U-Net model into a real-time audio pipeline and a web application.

---

## Key Features

- Audio preprocessing and normalization
- 16 kHz mono speech processing
- STFT-based spectrogram representation
- Synthetic noisy speech generation at multiple SNR levels
- Spectral subtraction baseline
- CNN-based speech enhancement
- U-Net-based speech enhancement
- Mask U-Net-based speech enhancement
- SNR evaluation
- SI-SDR evaluation
- STOI speech intelligibility evaluation
- Model comparison
- Real-time microphone-based enhancement
- FastAPI web application
- Audio upload and enhancement
- Enhanced audio playback
- Enhanced audio download

---

## System Architecture

```text
                    NOISY SPEECH
                         |
                         v
                Audio Preprocessing
                         |
                         v
                        STFT
                         |
                         v
               Noisy Spectrogram
                         |
                         v
                 Mask U-Net Model
                         |
                         v
                Predicted Speech Mask
                         |
                         v
        Noisy Magnitude × Speech Mask
                         |
                         v
                Enhanced Magnitude
                         |
                  +      |
                  | Noisy Phase
                  |
                         v
                       iSTFT
                         |
                         v
                  ENHANCED SPEECH
                         |
                         v
             SNR / SI-SDR / STOI
````

---

# Models Implemented

## 1. Spectral Subtraction

A traditional DSP-based noise reduction method is implemented as the initial baseline.

```
```

```
Noisy Speech
     |
     v
Noise Estimation
     |
     v
Spectral Subtraction
     |
     v
Enhanced Spectrogram
     |
     v
iSTFT
     |
     v
Enhanced Speech
```

The implementation estimates noise from the beginning of the signal and subtracts the estimated noise spectrum from the noisy spectrum.

---

## 2. CNN Baseline

A simple convolutional neural network was implemented to predict the clean speech log-magnitude spectrogram.

Architecture:

```
```

```
Input
  |
Conv2D
  |
ReLU
  |
Conv2D
  |
ReLU
  |
Conv2D
  |
ReLU
  |
Conv2D
  |
Output
```

This model provides a simple deep learning baseline for comparison.

---

## 3. U-Net

A U-Net architecture was implemented using an encoder-decoder structure with skip connections.

```
```

```
                 Input
                   |
              Encoder
                   |
             Bottleneck
                   |
              Decoder
                   |
                 Output

       Encoder ----------> Decoder
             Skip Connections
```

The skip connections allow the decoder to recover detailed time-frequency information from earlier layers.

Model size:

-  Parameters: 483,153 
-  Approximate FP32 size: 1.84 MB 

---

## 4. Mask U-Net

The final real-time model uses a Mask U-Net architecture.

Instead of directly predicting the clean spectrogram, the model predicts a speech mask between 0 and 1.

```
```

```
              Noisy Spectrogram
                      |
                      v
                  Mask U-Net
                      |
                      v
                Speech Mask
                  [0, 1]
                      |
                      v
       Noisy Magnitude × Speech Mask
                      |
                      v
             Enhanced Magnitude
                      |
                      v
                    iSTFT
                      |
                      v
              Enhanced Speech
```

Model size:

-  Parameters: 483,153 
-  Approximate FP32 size: 1.84 MB 

The trained checkpoint used by the real-time and web applications is:

```
```

```
checkpoints/mask_unet_best.pth
```

---

# Dataset

The project uses a controlled speech enhancement dataset generated from clean speech and different noise sources.

### Noise types

-  White noise 
-  Low-frequency noise 
-  Mixed noise 

### SNR levels

```
```

```
-5 dB
 0 dB
+5 dB
+10 dB
```

### Current dataset split

```
```

```
Training recordings:      8
Validation recordings:    1
Test recordings:          2
```

Generated noisy examples:

```
```

```
Training:      32
Validation:     4
Testing:        8
```

The current dataset is intentionally small and is primarily intended for demonstrating the complete speech enhancement pipeline.

For production-level evaluation, a larger multi-speaker dataset should be used.

---

# Audio Processing

All audio is processed using:

```
```

```
Sample Rate: 16,000 Hz
Channels:    Mono
Format:      WAV
```

The STFT configuration is:

```
```

```
N_FFT:       512
Hop Length:  128
Window:      Hann
Window Size: 512
```

The magnitude spectrogram is converted into log scale:

```
```

```
log_magnitude = log(1 + magnitude)
```

This representation is used as the input to the neural networks.

---

# Training

The models are trained using PyTorch.

The training pipeline performs:

```
```

```
Noisy Audio
     |
     v
STFT
     |
     v
Log Magnitude
     |
     v
Neural Network
     |
     v
Predicted Spectrogram / Mask
     |
     v
Loss Calculation
     |
     v
Backpropagation
     |
     v
Model Update
```

The primary training loss is Mean Squared Error (MSE) between predicted and target log-magnitude spectrograms.

---

# Evaluation Metrics

Three metrics are used to evaluate speech enhancement.

## SNR

Signal-to-Noise Ratio measures the relative strength of the desired signal compared with noise.

Higher improvement indicates greater noise reduction.

---

## SI-SDR

Scale-Invariant Signal-to-Distortion Ratio evaluates the quality of the reconstructed speech while being less sensitive to overall signal scaling.

Higher values generally indicate better reconstruction.

---

## STOI

Short-Time Objective Intelligibility measures estimated speech intelligibility.

STOI values generally range from 0 to 1, with higher values indicating greater estimated intelligibility.

---

# Experimental Results

The following results were obtained on the current test set.

| ModelAverage SNR ImprovementAverage SI-SDR ImprovementAverage STOI Improvement |          |          |        |
| ------------------------------------------------------------------------------ | -------- | -------- | ------ |
| CNN                                                                            | +0.72 dB | +0.34 dB | -0.040 |
| U-Net                                                                          | +5.19 dB | +5.22 dB | -0.006 |
| Mask U-Net                                                                     | +5.55 dB | +5.42 dB | -0.016 |

These values are experimental results on the current small test dataset and should not be interpreted as general performance guarantees.

The complete model comparison is stored in:

```
```

```
outputs/metrics/all_models_comparison.csv
```

---

# Real-Time Speech Enhancement

The project includes a real-time microphone enhancement prototype.

Architecture:

```
```

```
Microphone
    |
    v
Audio Stream
    |
    v
Input Queue
    |
    v
Worker Thread
    |
    v
STFT
    |
    v
Mask U-Net
    |
    v
Speech Mask
    |
    v
Enhanced Magnitude
    |
    v
iSTFT
    |
    v
Output Queue
    |
    v
Speaker / Headphones
```

The real-time implementation uses a threaded queue architecture so that microphone input/output callbacks remain lightweight while neural network inference runs in a separate worker thread.

Run:

```
```

```
python realtime/realtime_enhancer.py
```

The application uses the trained Mask U-Net checkpoint:

```
```

```
checkpoints/mask_unet_best.pth
```

---

# Web Application

A FastAPI-based web application is included.

The web application allows users to:

1.  Open the application in a browser 
2.  Upload a noisy audio file 
3.  Send the audio to the FastAPI backend 
4.  Run Mask U-Net inference 
5.  Generate enhanced speech 
6.  Play the enhanced audio 
7.  Download the enhanced WAV file 

Architecture:

```
```

```
                 Browser
                    |
                    v
          HTML / CSS / JavaScript
                    |
                    v
              FastAPI API
                    |
                    v
             Speech Enhancer
                    |
                    v
               Mask U-Net
                    |
                    v
                 iSTFT
                    |
                    v
             Enhanced WAV
                    |
                    v
                 Browser
```

---

# Running the Web Application

Activate the virtual environment:

```
```

```
.\venv\Scripts\Activate.ps1
```

Start the FastAPI server:

```
```

```
python -m uvicorn webapp.app:app --reload
```

Open:

```
```

```
http://127.0.0.1:8000
```

The application provides a simple interface for uploading and enhancing audio.

---

# Project Structure

```
```

```
speech enhancement and noise suppression/
│
├── configs/
│   └── config.py
│
├── data/
│   ├── metadata/
│   ├── processed/
│   ├── raw/
│   │   ├── clean/
│   │   └── noise/
│   ├── train/
│   │   ├── clean/
│   │   └── noisy/
│   ├── validation/
│   │   ├── clean/
│   │   └── noisy/
│   └── test/
│       ├── clean/
│       └── noisy/
│
├── outputs/
│   ├── audio/
│   ├── metrics/
│   └── spectrograms/
│
├── checkpoints/
│
├── src/
│   ├── audio/
│   │   ├── preprocessing.py
│   │   ├── utils.py
│   │   ├── stft.py
│   │   └── spectral_subtraction.py
│   │
│   ├── data/
│   │   ├── mixing.py
│   │   ├── dataset_split.py
│   │   ├── speech_dataset.py
│   │   └── spectrogram_dataset.py
│   │
│   ├── models/
│   │   ├── cnn_baseline.py
│   │   ├── unet.py
│   │   └── mask_unet.py
│   │
│   └── training/
│       ├── loss.py
│       ├── train.py
│       ├── train_unet.py
│       └── train_mask_unet.py
│
├── realtime/
│   └── realtime_enhancer.py
│
├── webapp/
│   ├── app.py
│   ├── inference.py
│   ├── static/
│   │   ├── style.css
│   │   └── script.js
│   └── templates/
│       └── index.html
│
├── requirements.txt
│
├── generate_noise_files.py
├── generate_noisy_dataset.py
├── download_speech_dataset.py
├── split_dataset.py
│
├── test_audio_pipeline.py
├── test_stft.py
├── test_dataset.py
├── test_model.py
│
├── run_spectral_subtraction.py
├── run_cnn_inference.py
├── run_unet_inference.py
├── run_mask_unet_inference.py
│
├── evaluate_cnn.py
├── evaluate_unet.py
├── evaluate_mask_unet.py
├── compare_cnn_unet.py
├── compare_all_models.py
│
└── README.md
```

---

# Installation

## 1. Clone the repository

```
```

```
git clone https://github.com/NikhithaKarrolla/speech-enhancement-and-noise-suppression.git
```

```
```

```
cd speech-enhancement-and-noise-suppression
```

> Replace the repository URL above with the actual GitHub repository URL if the repository name is different.

---

## 2. Create a virtual environment

Windows:

```
```

```
python -m venv venv
```

Activate:

```
```

```
.\venv\Scripts\Activate.ps1
```

---

## 3. Install dependencies

```
```

```
pip install -r requirements.txt
```

---

# Usage

## Test the audio pipeline

```
```

```
python test_audio_pipeline.py
```

## Test STFT

```
```

```
python test_stft.py
```

## Test the dataset

```
```

```
python test_dataset.py
```

## Test the models

```
```

```
python test_model.py
```

## Run real-time enhancement

```
```

```
python realtime/realtime_enhancer.py
```

## Run web application

```
```

```
python -m uvicorn webapp.app:app --reload
```

Then open:

```
```

```
http://127.0.0.1:8000
```

---

# Technologies Used

### Programming

-  Python 
-  JavaScript 
-  HTML 
-  CSS 

### Deep Learning

-  PyTorch 
-  CNN 
-  U-Net 
-  Mask U-Net 

### Audio Processing

-  Librosa 
-  SoundFile 
-  NumPy 
-  SciPy 
-  SoundDevice 

### Evaluation

-  SNR 
-  SI-SDR 
-  STOI 

### Web

-  FastAPI 
-  Uvicorn 
-  Jinja2 

### Development

-  Git 
-  GitHub 
-  VS Code 

---

# Limitations

The current implementation is a research/academic prototype.

### Dataset size

The current dataset contains a small number of recordings. Therefore, the reported metrics are specific to the current test set.

### Validation set

Only one clean recording is currently used for validation. A larger validation set would provide a more reliable estimate of generalization.

### Noise types

The current experiments use controlled noise types rather than a large collection of real-world environmental noises.

### Real-time latency

The real-time implementation currently performs neural network inference on the local CPU when GPU acceleration is unavailable. This may introduce latency depending on the hardware.

### Noisy phase

The reconstruction currently uses the noisy phase rather than estimating a clean phase.

### Web application

The current web application performs upload-based enhancement. Continuous browser microphone streaming can be added as a future extension.

---

# Future Improvements

Potential improvements include:

-  Train on a much larger multi-speaker speech dataset 
-  Add real-world environmental noise recordings 
-  Improve real-time latency 
-  Add browser microphone recording 
-  Implement WebSocket/WebRTC streaming 
-  Estimate clean phase 
-  Use multi-resolution STFT loss 
-  Add perceptual audio losses 
-  Experiment with CRNN and Transformer-based architectures 
-  Export the model to ONNX 
-  Apply quantization 
-  Optimize inference for edge devices 
-  Deploy the web application to a cloud platform 
-  Add authentication and user accounts 
-  Add visual spectrogram comparison 
-  Add downloadable evaluation reports 

---

# Learning Outcomes

This project demonstrates practical knowledge of:

-  Digital signal processing 
-  STFT and iSTFT 
-  Spectrogram analysis 
-  Noise generation and SNR 
-  CNN architectures 
-  Encoder-decoder networks 
-  U-Net skip connections 
-  Mask-based speech enhancement 
-  PyTorch model training 
-  Model evaluation 
-  Real-time audio streaming 
-  Multithreaded audio processing 
-  FastAPI backend development 
-  Frontend-backend integration 
-  End-to-end machine learning application development 

---

# Author

**Nikhitha Karrolla**

Computer Science and Engineering

GitHub:

https://github.com/NikhithaKarrolla