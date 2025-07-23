# ⚡ Power Quality Segmentation Pipeline

This project is a deep learning-based segmentation pipeline for analyzing power quality data. It uses Docker for containerized deployment and includes options for both GPU and CPU execution.

---

## 📦 Project Overview

The pipeline includes:
- Python scripts and Jupyter notebooks
- Dockerized backend and environment
- Frontend served with FastAPI
- Support for GPU (via NVIDIA Docker)

---

## ✅ Prerequisites

Make sure the following are installed:

- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (if using GPU)
- Git

Optional (for local Python runs):
- Python 3.8+ and `pip`

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/zubayer16/thesis-ecophi-zubayer.git
cd power-quality-segmentation_Fault
```


## 2. 🚀 Run the Pipeline (Recommended)
To launch the full pipeline (build + run):
```bash
uvicorn run_server:app --reload
```

[ALTERNATIVE OPTIONS] For running docker compose and building an image you can run this script:

```bash
.\docker_build_run.bat
```
- Checks Docker/NVIDIA status
- Creates necessary directories
- Prompts for running options:
```bash
  Choose running option:
1) Build and run with GPU support
2) Build and run without GPU support
3) Run with Docker Compose
4) Just build the image
```
Choose based on your system (e.g., 1 for GPU, 2 for CPU)

## 3. 🖥️ User Interface Overview
The frontend of the Power Quality Segmentation Pipeline provides an intuitive, interactive dashboard for exploring model results. It is served via FastAPI and is accessible through your browser at:
```bash
http://localhost:8000
```
![UI Screenshot](images/ui_dashboard.png)
## 🔧 Configuration Panel (Left Sidebar)
The configuration panel lets you customize your visualization and analysis:

- Phase Type: Choose between voltage or current data for analysis (e.g., Current, Voltage).

- Phase Number: Select the specific phase (e.g., Phase 1, Phase 2, Phase 3).

- Number of Samples: Enter the number of time series samples to visualize (valid range: 10–500).

- Visualization Type: Select between different visualization modes, such as:
  - Comparison Dashboard (default)
  - Single View
- Generate Visualization: Click this button to update and regenerate the results view with the selected settings.

- Download Results: Export segmentation results and predictions in .csv or .json format.

- Quick Stats:Inference Time: Time taken to segment a single sample.

- Throughput: Number of samples processed per second.

- Device: Indicates whether the CPU or GPU was used for inference.

## Training Setup Overview
The training process runs for approximately 10 minutes (CPU), utilizing a GRU (Gated Recurrent Unit) combined with an Attention mechanism. The model is trained sequentially on individual phases of the data, allowing it to capture temporal dependencies and focus on the most relevant parts of the sequence for each phase. The following steps are performed while training

- STEP 1: INITIALIZING TimeSeriesSegmentation CLASS
- STEP 2: PREPARING DATA (LOAD & PREPROCESS)
- STEP 3: PREPARING TRAINING DATA (SPLIT & SEQUENCE)
- STEP 4: MODEL INITIALIZATION
- SEQUENTIAL TRAINING - VOLTAGE & CURRENT PHASES

## 📁 Project Structure

| Path                     | Description                                  |
| ------------------------ | -------------------------------------------- |
| `app.py` / `fast_api.py` | FastAPI server entry points                  |
| `static/`                | HTML/CSS/JS files for frontend               |
| `models/`                | Trained model weights                        |
| `data/`                  | Segmentation input and label files           |
| `outputs/` / `logs/`     | Output predictions and log files             |
| `docker_build_run.bat`   | 🚀 Main script to build and run the pipeline |
| `Dockerfile`             | Docker build script                          |
| `docker-compose.yml`     | Docker Compose setup (if used)               |

