# cuda-powered-transcriber

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-green.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)
![Model](https://img.shields.io/badge/Model-Whisper--Large--V3-orange.svg)

A high-performance, GPU-accelerated video transcription tool built with **Faster-Whisper** and **Streamlit**. Made in under 4 hours of braincells and 20 grams of caffeine for a research. Utilizes 1.55B parameter model, so at the very least have a good machine.

---

## Installation
### 1. Simply clone the repository and then: 
```bash
pip install -r requirements.txt
```

### 2. The NVIDIA DLLs (Important!)
To run on GPU (CUDA), this thing requires specific NVIDIA libraries. The code is designed to find them automatically if they are installed via pip.

If you get a `cublas64_12.dll` error, simply download the missing DLLs and place them in the root folder:
- `cublas64_12.dll`
- `cublasLt64_12.dll`
- `cudnn64_8.dll`

### 3. Run the App
```bash
streamlit run transcriptor.py
```

---

## GPU verifier
This repo includes a `test_gpu.py` script. Run this to verify if your environment is correctly seeing your NVIDIA GPU:

```bash
python test_gpu.py
```

---

## How it works basically
1. Uses `MoviePy` to strip audio from your video files.
2. Loads the Whisper model using `float16` precision for the best balance of speed and VRAM usage.
3. Processes segments in real-time with timestamps.

---

## Requirements
- **Python**: 3.12+
- **GPU**: NVIDIA GPU with 6GB+ VRAM (for Large-v3)
- **Libraries**:
    - `faster-whisper`
    - `streamlit`
    - `moviepy`
    - `nvidia-cublas-cu12` (for CUDA 12 support)
