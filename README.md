Based on the structure of the DL Streamer workshop example from the GitHub link and the contents of your attached document (`Report-Intel.docx`), here's a formatted lab guide for your project titled **"Scalable AI-based Video Analytics Pipeline using DL Streamer on Intel Hardware"** — structured like the Intel workshop documentation.

---

# **Build a Scalable AI-based Video Analytics Pipeline using DL Streamer**

## **Lab Objectives**

In this lab, you will:

* Build a video analytics pipeline using DL Streamer.
* Deploy and test the pipeline on Intel CPU and GPU hardware.
* Measure performance across multiple streams.
* Analyze bottlenecks and recommend optimizations.

---

## **1. Introduction**

Modern city surveillance systems often use numerous cameras to monitor events like **Mahakumbh 2025** or **international tournaments**. Manual monitoring of such streams is inefficient and unscalable.

To address this challenge, we leverage Intel’s **DL Streamer** and **OpenVINO Toolkit** to build a pipeline that performs:

* Video decoding
* Object detection
* Object classification

---

## **2. Prerequisites**

### **Hardware**

* Intel Core i7 CPU or higher
* Intel Iris Xe / Xeon GPU (recommended)

### **Software**

* Ubuntu 20.04 or later
* Python 3.8+
* DL Streamer ([https://github.com/dlstreamer/dlstreamer](https://github.com/dlstreamer/dlstreamer))
* OpenVINO Toolkit 2022.3 or later
* GStreamer
* VAAPI / Intel Media SDK

### **Install Dependencies**

```bash
# Install GStreamer
sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-base \
gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
gstreamer1.0-plugins-ugly gstreamer1.0-libav

# Clone and install DL Streamer
git clone https://github.com/dlstreamer/dlstreamer.git
cd dlstreamer
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

---

## **3. Pipeline Architecture**

### **Flow Overview**

```
rtspsrc ! decodebin ! gvadetect ! gvaclassify ! gvametaconvert ! gvametapublish
```

### **Stages**

| Stage        | Description                                                         |
| ------------ | ------------------------------------------------------------------- |
| **Decode**   | Input from RTSP/MP4, decode with H.264/H.265 via VAAPI or Media SDK |
| **Detect**   | Object detection using models like YOLOv8 or SSD                    |
| **Classify** | Adds metadata (e.g., age/gender, vehicle type)                      |

---

## **4. Setup Test Inputs**

Use sample RTSP streams or video files (`.mp4`) located locally:

```bash
export INPUT=videos/sample_input.mp4
```

---

## **5. Run the Pipeline**

### **Sample GStreamer Command**

```bash
gst-launch-1.0 filesrc location=$INPUT ! \
decodebin ! \
gvadetect model=ssd.xml device=CPU ! \
gvaclassify model=age-gender.xml device=CPU ! \
gvametaconvert ! \
gvametapublish method=file file-path=output.json ! \
fpsdisplaysink video-sink=xvimagesink sync=false
```

*Replace `ssd.xml` and `age-gender.xml` with OpenVINO IR models.*

---

## **6. Performance Evaluation**

| Hardware            | Streams | FPS (avg) | Model            | Bottleneck       |
| ------------------- | ------- | --------- | ---------------- | ---------------- |
| Intel Core i7 CPU   | 2       | \~18      | SSD              | CPU              |
| Intel GPU (Iris Xe) | 4       | \~25      | YOLOv8-tiny      | Memory Bandwidth |
| Intel Xeon + iGPU   | 6       | \~30      | SSD + Classifier | I/O              |

### **Test Parameters**

* Vary the number of input streams (1 to 6).
* Compare performance of SSD vs. YOLOv8-tiny models.
* Measure FPS using `fpsdisplaysink` or custom metrics.

---

## **7. Scalability Analysis**

| Factor         | Observation                                                              |
| -------------- | ------------------------------------------------------------------------ |
| **Compute**    | GPUs outperform CPUs for concurrent streams.                             |
| **Memory**     | Classifiers are memory-intensive. Tune batch size.                       |
| **I/O**        | Disk/Network I/O becomes a bottleneck >4 streams.                        |
| **Model Type** | Lightweight models offer better throughput with minor accuracy tradeoff. |

---

## **8. Use Case Applications**

### **Mahakumbh 2025**

* Real-time face detection.
* Crowd behavior analysis for safety.

### **ICC Tournaments**

* Detect players and fans.
* Highlight generation for events.

---

## **9. Recommendations**

* For scalability, use **YOLOv8-tiny** + **OpenVINO** on **Intel GPU**.
* Explore:

  * Object **tracking modules** (e.g., DeepSORT).
  * **Asynchronous inference** for better performance.
  * **Edge-to-cloud** integration using **Kafka/MQTT**.

---

## **10. Resources**

* [DL Streamer GitHub](https://github.com/dlstreamer/dlstreamer)
* [OpenVINO Toolkit](https://docs.openvino.ai/)
* Intel DevCloud or Intel AI Reference Kits

---

Would you like this as a formatted **Markdown file**, **HTML**, or **Word document** for your report submission or GitHub documentation?
