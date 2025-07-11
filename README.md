# Scalable AI-based Video Analytics Pipeline using DL Streamer

This repository demonstrates how to build a real-time, multi-stream video analytics pipeline using **Intel® DL Streamer** and **OpenVINO™ Toolkit**. The project supports object detection and classification on Intel CPU and GPU hardware, with use cases tailored to smart city surveillance such as **Mahakumbh 2025** and **ICC Tournaments**.

# Project By-
[**Tanish Das**](https://github.com/dastanish2256) (tdas2@gitam.in)
[**Piyush Ram Kimidi**](https://github.com/piyushram612)(pkimidi@gitam.in) 
[**Harshith Kakkireni**](https://github.com/kakkireni-harshith) (hkakkire@gitam.in)

---

## 🚩 Table of Contents

* [Project Overview](#project-overview)
* [Objectives](#objectives)
* [System Requirements](#system-requirements)
* [Environment Setup](#environment-setup)
* [Pipeline Architecture](#pipeline-architecture)
* [Running the Pipeline](#running-the-pipeline)
* [Performance Evaluation](#performance-evaluation)
* [Scalability Analysis](#scalability-analysis)
* [Use Case Scenarios](#use-case-scenarios)
* [Enhancements & Recommendations](#enhancements--recommendations)
* [References](#references)

---

## 📘 Project Overview

Modern cities and event venues require scalable AI video analytics for real-time monitoring. Manual camera feed analysis is inefficient when dealing with dozens or hundreds of streams. This project implements a **GStreamer-based pipeline** capable of decoding, detecting, and classifying objects using OpenVINO-optimized models, running across Intel hardware.

---

## 🎯 Objectives

* Build a modular DL Streamer pipeline: decode → detect → classify → publish.
* Evaluate performance across CPU, GPU, and hybrid configurations.
* Identify and address hardware and software bottlenecks.
* Demonstrate scalability with multi-stream testing.

---

## 💻 System Requirements

### Hardware

* Intel Core i7 CPU (minimum)
* Intel Iris Xe, HD Graphics, or Xeon iGPU (recommended)

### Software

* Ubuntu 20.04 or later
* Python 3.8+
* GStreamer 1.18+
* OpenVINO 2022.3+
* DL Streamer
* Intel Media SDK or VAAPI

---

## ⚙️ Environment Setup

### 1. Install Dependencies

```bash
sudo apt update && sudo apt install -y \
build-essential cmake git python3-pip \
gstreamer1.0-tools gstreamer1.0-plugins-base \
gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
gstreamer1.0-plugins-ugly gstreamer1.0-libav \
libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
pkg-config xorg-dev
```

### 2. Install OpenVINO Toolkit

Follow [OpenVINO installation guide](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html):

```bash
source /opt/intel/openvino_2022/setupvars.sh
```

### 3. Build DL Streamer

```bash
git clone https://github.com/dlstreamer/dlstreamer.git
cd dlstreamer
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

---

## 🧩 Pipeline Architecture

### Functional Stages

1. **Decode** – Handles MP4/RTSP input decoding using VAAPI or Media SDK.
2. **Detect** – Runs object detection (e.g., YOLOv8, SSD).
3. **Classify** – Classifies detected entities (e.g., Age-Gender, Vehicle Type).
4. **Convert & Publish** – Converts metadata and exports via file, Kafka, or console.

### GStreamer Flow

```
rtspsrc / filesrc
    → decodebin
    → gvadetect
    → gvaclassify
    → gvametaconvert
    → gvametapublish
```

---

## ▶️ Running the Pipeline

### A. Run on Local Video File (SSD model, CPU)

```bash
gst-launch-1.0 filesrc location=video.mp4 ! \
decodebin ! \
gvadetect model=ssd.xml model_proc=ssd.json device=CPU ! \
gvaclassify model=age-gender.xml model_proc=age-gender.json device=CPU ! \
gvametaconvert format=json ! \
gvametapublish method=file file-path=output.json ! \
fpsdisplaysink video-sink=xvimagesink sync=false
```

### B. Run on RTSP Stream (YOLOv8-tiny, GPU)

```bash
gst-launch-1.0 rtspsrc location=rtsp://<stream_url> latency=100 ! \
rtph264depay ! h264parse ! vaapidecodebin ! \
gvadetect model=yolov8-tiny.xml model_proc=yolov8.json device=GPU ! \
gvaclassify model=vehicle-type.xml model_proc=vehicle-type.json device=GPU ! \
gvametaconvert format=json ! \
gvametapublish method=console ! \
fpsdisplaysink video-sink=xvimagesink sync=false
```

---

## 📊 Performance Evaluation

| Hardware          | Streams | Avg FPS | Model            | Bottleneck         |
| ----------------- | ------- | ------- | ---------------- | ------------------ |
| Intel Core i7 CPU | 2       | \~18    | SSD              | CPU                |
| Intel Iris Xe GPU | 4       | \~25    | YOLOv8-tiny      | Memory Bandwidth   |
| Intel Xeon + iGPU | 6       | \~30    | SSD + Classifier | I/O (disk/network) |

### Measuring FPS

```bash
fpsdisplaysink text-overlay=false video-sink=fakesink
```

Or use OpenVINO’s `benchmark_app` for offline model testing.

---

## 📈 Scalability Analysis

| Factor       | Impact                                                                |
| ------------ | --------------------------------------------------------------------- |
| Compute      | Intel GPUs outperform CPUs for high stream concurrency.               |
| Memory       | Classifiers are memory-intensive; batching helps improve performance. |
| I/O          | Disk or RTSP latency limits FPS with >4 streams on CPU.               |
| Model Choice | YOLOv8-tiny offers best speed/accuracy tradeoff.                      |

---

## 🧠 Use Case Scenarios

### Mahakumbh 2025 Surveillance

* Detect crowd density and facial activity.
* Classify age-gender patterns in real time.

### ICC Tournament Monitoring

* Track players and fans across multiple zones.
* Generate automated highlights based on detected events.

---

## 🔧 Enhancements & Recommendations

* Add **object tracking** (e.g., DeepSORT or BYTETrack).
* Enable **asynchronous inference** for better throughput.
* Integrate with **MQTT** or **Kafka** for edge-cloud communication.
* Tune pipeline with **batch size**, **buffer pool**, and **async mode**.

---

## 📚 References

* [DL Streamer GitHub](https://github.com/dlstreamer/dlstreamer)
* [OpenVINO Documentation](https://docs.openvino.ai/)
* [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo)
* Mahakumbh 2025 AI Planning Reports
* ICC Digital Innovation Whitepapers

---

