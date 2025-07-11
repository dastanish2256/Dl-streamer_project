# DL Streamer Pipeline Project: Complete Implementation Guide

## Project Overview
Create a scalable video analytics pipeline using Intel's DL Streamer framework that can:
- **Detect** objects in video streams
- **Decode** video content efficiently
- **Classify** detected objects
- Analyze system scalability on Intel hardware (CPU and GPU)

## Phase 1: Environment Setup

### 1.1 Prerequisites Installation
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker (recommended approach for DL Streamer)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# Logout and login again

# Install git and other dependencies
sudo apt install git wget curl python3 python3-pip -y
```

### 1.2 Install Intel DL Streamer via Docker
```bash
# Pull the latest DL Streamer Docker image
docker pull intel/dlstreamer:latest

# For Ubuntu 22.04 specifically:
# docker pull intel/dlstreamer:2025.0.1.2-ubuntu22

# Verify installation
docker run --rm intel/dlstreamer:latest gst-inspect-1.0 gvadetect
```

### 1.3 Setup Intel Hardware Acceleration
```bash
# Install Intel GPU drivers (if available)
sudo apt install intel-gpu-tools vainfo
vainfo  # Check hardware video acceleration support

# For Intel iGPU support
sudo apt install intel-media-va-driver-non-free
```

## Phase 2: Project Structure Setup

### 2.1 Create Project Directory
```bash
mkdir dl_streamer_project
cd dl_streamer_project
mkdir -p {models,videos,scripts,results,reports}
```

### 2.2 Download Pre-trained Models
```bash
# Create model download script
cat > scripts/download_models.py << 'EOF'
import os
import urllib.request
import json

def download_model(model_name, model_url, model_dir):
    """Download OpenVINO model files"""
    os.makedirs(model_dir, exist_ok=True)
    
    # Download .xml and .bin files
    for ext in ['.xml', '.bin']:
        url = f"{model_url}/{model_name}{ext}"
        filepath = os.path.join(model_dir, f"{model_name}{ext}")
        
        if not os.path.exists(filepath):
            print(f"Downloading {model_name}{ext}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded to {filepath}")

# Common OpenVINO models
models = {
    "face-detection-adas-0001": "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/1/face-detection-adas-0001/FP32",
    "person-detection-retail-0013": "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/1/person-detection-retail-0013/FP32",
    "age-gender-recognition-retail-0013": "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/1/age-gender-recognition-retail-0013/FP32"
}

for model_name, base_url in models.items():
    download_model(model_name, base_url, "models")
EOF

python3 scripts/download_models.py
```

### 2.3 Prepare Test Videos
```bash
# Download sample video
wget -O videos/test_video.mp4 "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4"

# Create multiple test videos with different resolutions
ffmpeg -i videos/test_video.mp4 -s 640x480 -t 30 videos/test_480p.mp4
ffmpeg -i videos/test_video.mp4 -s 1920x1080 -t 30 videos/test_1080p.mp4
```

## Phase 3: Pipeline Implementation

### 3.1 Basic Detection Pipeline
```bash
# Create basic detection script
cat > scripts/basic_detection.py << 'EOF'
#!/usr/bin/env python3
import subprocess
import sys
import time
import json
import os

def run_detection_pipeline(input_video, model_path, device="CPU", batch_size=1):
    """Run basic detection pipeline"""
    
    pipeline_cmd = [
        "gst-launch-1.0",
        f"filesrc location={input_video} !",
        "decodebin !",
        "videoconvert !",
        f"gvadetect model={model_path} device={device} batch-size={batch_size} !",
        "gvawatermark !",
        "videoconvert !",
        "fpsdisplaysink video-sink=fakesink text-overlay=false"
    ]
    
    cmd = " ".join(pipeline_cmd)
    print(f"Running: {cmd}")
    
    start_time = time.time()
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    end_time = time.time()
    
    return {
        "duration": end_time - start_time,
        "stdout": stdout.decode(),
        "stderr": stderr.decode(),
        "return_code": process.returncode
    }

if __name__ == "__main__":
    # Test with person detection model
    result = run_detection_pipeline(
        "videos/test_video.mp4",
        "models/person-detection-retail-0013.xml",
        device="CPU"
    )
    
    print(f"Pipeline completed in {result['duration']:.2f} seconds")
    if result['return_code'] != 0:
        print(f"Error: {result['stderr']}")
EOF

chmod +x scripts/basic_detection.py
```

### 3.2 Complete Detection + Classification Pipeline
```bash
cat > scripts/full_pipeline.py << 'EOF'
#!/usr/bin/env python3
import subprocess
import sys
import time
import json
import psutil
import threading
from collections import defaultdict

class PipelineRunner:
    def __init__(self):
        self.performance_data = defaultdict(list)
        self.monitoring = False
        
    def monitor_resources(self):
        """Monitor CPU and GPU usage during pipeline execution"""
        while self.monitoring:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            self.performance_data['cpu'].append(cpu_percent)
            self.performance_data['memory'].append(memory_percent)
            
            # Try to get GPU info if available
            try:
                gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits']).decode().strip()
                if gpu_info:
                    gpu_util, gpu_mem = gpu_info.split(',')
                    self.performance_data['gpu'].append(float(gpu_util))
                    self.performance_data['gpu_memory'].append(float(gpu_mem))
            except:
                pass
                
            time.sleep(1)
    
    def run_pipeline(self, input_video, detection_model, classification_model, device="CPU", batch_size=1):
        """Run complete detection + classification pipeline"""
        
        pipeline_cmd = [
            "gst-launch-1.0",
            f"filesrc location={input_video} !",
            "decodebin !",
            "videoconvert !",
            f"gvadetect model={detection_model} device={device} batch-size={batch_size} !",
            f"gvaclassify model={classification_model} device={device} batch-size={batch_size} !",
            "gvawatermark !",
            "videoconvert !",
            "fpsdisplaysink video-sink=fakesink text-overlay=false"
        ]
        
        cmd = " ".join(pipeline_cmd)
        print(f"Running pipeline on {device}: {cmd}")
        
        # Start monitoring
        self.monitoring = True
        monitor_thread = threading.Thread(target=self.monitor_resources)
        monitor_thread.start()
        
        start_time = time.time()
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        end_time = time.time()
        
        # Stop monitoring
        self.monitoring = False
        monitor_thread.join()
        
        return {
            "duration": end_time - start_time,
            "device": device,
            "batch_size": batch_size,
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
            "return_code": process.returncode,
            "performance": dict(self.performance_data)
        }

def test_scalability():
    """Test pipeline scalability with different configurations"""
    runner = PipelineRunner()
    
    test_configs = [
        {"device": "CPU", "batch_size": 1},
        {"device": "CPU", "batch_size": 2},
        {"device": "CPU", "batch_size": 4},
        {"device": "GPU", "batch_size": 1},
        {"device": "GPU", "batch_size": 2},
        {"device": "GPU", "batch_size": 4},
    ]
    
    results = []
    
    for config in test_configs:
        try:
            result = runner.run_pipeline(
                "videos/test_video.mp4",
                "models/person-detection-retail-0013.xml",
                "models/age-gender-recognition-retail-0013.xml",
                device=config["device"],
                batch_size=config["batch_size"]
            )
            results.append(result)
            
            # Save intermediate results
            with open(f"results/result_{config['device']}_batch{config['batch_size']}.json", "w") as f:
                json.dump(result, f, indent=2)
                
        except Exception as e:
            print(f"Error with config {config}: {e}")
            
    return results

if __name__ == "__main__":
    results = test_scalability()
    
    # Save all results
    with open("results/scalability_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Scalability testing completed!")
EOF

chmod +x scripts/full_pipeline.py
```

### 3.3 Multi-Stream Testing
```bash
cat > scripts/multi_stream_test.py << 'EOF'
#!/usr/bin/env python3
import subprocess
import threading
import time
import json
from concurrent.futures import ThreadPoolExecutor
import psutil

class MultiStreamTester:
    def __init__(self):
        self.results = []
        self.lock = threading.Lock()
        
    def run_single_stream(self, stream_id, input_video, device="CPU"):
        """Run a single stream pipeline"""
        pipeline_cmd = [
            "gst-launch-1.0",
            f"filesrc location={input_video} !",
            "decodebin !",
            "videoconvert !",
            f"gvadetect model=models/person-detection-retail-0013.xml device={device} !",
            "gvawatermark !",
            "videoconvert !",
            "fakesink"
        ]
        
        cmd = " ".join(pipeline_cmd)
        
        start_time = time.time()
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        end_time = time.time()
        
        with self.lock:
            self.results.append({
                "stream_id": stream_id,
                "duration": end_time - start_time,
                "device": device,
                "success": process.returncode == 0,
                "error": stderr.decode() if process.returncode != 0 else None
            })
        
        return process.returncode == 0
    
    def test_concurrent_streams(self, num_streams, device="CPU"):
        """Test multiple concurrent streams"""
        print(f"Testing {num_streams} concurrent streams on {device}")
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_streams) as executor:
            futures = []
            for i in range(num_streams):
                future = executor.submit(
                    self.run_single_stream,
                    i,
                    "videos/test_video.mp4",
                    device
                )
                futures.append(future)
            
            # Wait for all streams to complete
            success_count = sum(1 for future in futures if future.result())
        
        end_time = time.time()
        
        return {
            "num_streams": num_streams,
            "device": device,
            "total_time": end_time - start_time,
            "successful_streams": success_count,
            "failed_streams": num_streams - success_count,
            "individual_results": self.results.copy()
        }

def find_max_streams():
    """Find maximum number of streams for CPU and GPU"""
    tester = MultiStreamTester()
    
    results = {}
    
    # Test CPU scalability
    for device in ["CPU", "GPU"]:
        print(f"\nTesting {device} scalability...")
        device_results = []
        
        for num_streams in [1, 2, 4, 8, 16]:
            tester.results.clear()  # Reset results
            
            result = tester.test_concurrent_streams(num_streams, device)
            device_results.append(result)
            
            print(f"{device} - {num_streams} streams: {result['successful_streams']}/{num_streams} successful")
            
            # Stop if we start seeing failures
            if result['failed_streams'] > 0:
                print(f"Maximum streams for {device}: {num_streams - 1}")
                break
        
        results[device] = device_results
    
    return results

if __name__ == "__main__":
    results = find_max_streams()
    
    with open("results/multi_stream_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nMulti-stream testing completed!")
EOF

chmod +x scripts/multi_stream_test.py
```

## Phase 4: Performance Analysis & Benchmarking

### 4.1 Create Performance Analysis Script
```bash
cat > scripts/analyze_performance.py << 'EOF'
#!/usr/bin/env python3
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def analyze_scalability_results():
    """Analyze and visualize scalability results"""
    
    # Load results
    with open("results/scalability_results.json", "r") as f:
        results = json.load(f)
    
    # Create DataFrame for analysis
    data = []
    for result in results:
        if result['return_code'] == 0:  # Only successful runs
            data.append({
                'device': result['device'],
                'batch_size': result['batch_size'],
                'duration': result['duration'],
                'avg_cpu': np.mean(result['performance']['cpu']) if result['performance']['cpu'] else 0,
                'avg_memory': np.mean(result['performance']['memory']) if result['performance']['memory'] else 0,
                'fps': 30 / result['duration']  # Assuming 30 second video
            })
    
    df = pd.DataFrame(data)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # FPS vs Batch Size
    sns.barplot(data=df, x='batch_size', y='fps', hue='device', ax=axes[0,0])
    axes[0,0].set_title('FPS vs Batch Size')
    axes[0,0].set_xlabel('Batch Size')
    axes[0,0].set_ylabel('FPS')
    
    # CPU Usage vs Device
    sns.boxplot(data=df, x='device', y='avg_cpu', ax=axes[0,1])
    axes[0,1].set_title('CPU Usage by Device')
    axes[0,1].set_xlabel('Device')
    axes[0,1].set_ylabel('Average CPU Usage (%)')
    
    # Memory Usage vs Device
    sns.boxplot(data=df, x='device', y='avg_memory', ax=axes[1,0])
    axes[1,0].set_title('Memory Usage by Device')
    axes[1,0].set_xlabel('Device')
    axes[1,0].set_ylabel('Average Memory Usage (%)')
    
    # Processing Time vs Batch Size
    sns.lineplot(data=df, x='batch_size', y='duration', hue='device', ax=axes[1,1])
    axes[1,1].set_title('Processing Time vs Batch Size')
    axes[1,1].set_xlabel('Batch Size')
    axes[1,1].set_ylabel('Duration (seconds)')
    
    plt.tight_layout()
    plt.savefig('results/performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df

def analyze_multi_stream_results():
    """Analyze multi-stream performance"""
    
    with open("results/multi_stream_results.json", "r") as f:
        results = json.load(f)
    
    # Create summary report
    report = []
    
    for device, device_results in results.items():
        for result in device_results:
            report.append({
                'device': device,
                'num_streams': result['num_streams'],
                'successful_streams': result['successful_streams'],
                'success_rate': result['successful_streams'] / result['num_streams'],
                'total_time': result['total_time'],
                'avg_time_per_stream': result['total_time'] / result['num_streams']
            })
    
    df = pd.DataFrame(report)
    
    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Success rate vs number of streams
    sns.lineplot(data=df, x='num_streams', y='success_rate', hue='device', ax=axes[0])
    axes[0].set_title('Success Rate vs Number of Streams')
    axes[0].set_xlabel('Number of Streams')
    axes[0].set_ylabel('Success Rate')
    
    # Average processing time per stream
    sns.lineplot(data=df, x='num_streams', y='avg_time_per_stream', hue='device', ax=axes[1])
    axes[1].set_title('Average Processing Time per Stream')
    axes[1].set_xlabel('Number of Streams')
    axes[1].set_ylabel('Average Time (seconds)')
    
    plt.tight_layout()
    plt.savefig('results/multi_stream_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df

def generate_report():
    """Generate comprehensive performance report"""
    
    scalability_df = analyze_scalability_results()
    multi_stream_df = analyze_multi_stream_results()
    
    # Generate summary statistics
    report = {
        'scalability_summary': {
            'best_cpu_performance': scalability_df[scalability_df['device'] == 'CPU']['fps'].max(),
            'best_gpu_performance': scalability_df[scalability_df['device'] == 'GPU']['fps'].max() if 'GPU' in scalability_df['device'].values else 0,
            'optimal_cpu_batch': scalability_df[scalability_df['device'] == 'CPU'].loc[scalability_df[scalability_df['device'] == 'CPU']['fps'].idxmax(), 'batch_size'],
            'optimal_gpu_batch': scalability_df[scalability_df['device'] == 'GPU'].loc[scalability_df[scalability_df['device'] == 'GPU']['fps'].idxmax(), 'batch_size'] if 'GPU' in scalability_df['device'].values else 0,
        },
        'multi_stream_summary': {
            'max_cpu_streams': multi_stream_df[multi_stream_df['device'] == 'CPU']['successful_streams'].max(),
            'max_gpu_streams': multi_stream_df[multi_stream_df['device'] == 'GPU']['successful_streams'].max() if 'GPU' in multi_stream_df['device'].values else 0,
        }
    }
    
    with open('results/performance_summary.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

if __name__ == "__main__":
    # Install required packages
    import subprocess
    subprocess.run(["pip3", "install", "pandas", "matplotlib", "seaborn", "numpy"])
    
    # Generate analysis
    summary = generate_report()
    
    print("Performance Analysis Complete!")
    print(f"Best CPU Performance: {summary['scalability_summary']['best_cpu_performance']:.2f} FPS")
    print(f"Best GPU Performance: {summary['scalability_summary']['best_gpu_performance']:.2f} FPS")
    print(f"Max CPU Streams: {summary['multi_stream_summary']['max_cpu_streams']}")
    print(f"Max GPU Streams: {summary['multi_stream_summary']['max_gpu_streams']}")
EOF

chmod +x scripts/analyze_performance.py
```

## Phase 5: Docker Container Setup

### 5.1 Create Docker Run Script
```bash
cat > scripts/run_docker.sh << 'EOF'
#!/bin/bash

# Function to run pipeline in Docker container
run_pipeline_docker() {
    local VIDEO_PATH="$1"
    local DEVICE="$2"
    local BATCH_SIZE="$3"
    
    docker run --rm \
        --device /dev/dri \
        -v $(pwd):/workspace \
        -w /workspace \
        intel/dlstreamer:latest \
        gst-launch-1.0 \
        filesrc location=$VIDEO_PATH ! \
        decodebin ! \
        videoconvert ! \
        gvadetect model=models/person-detection-retail-0013.xml device=$DEVICE batch-size=$BATCH_SIZE ! \
        gvaclassify model=models/age-gender-recognition-retail-0013.xml device=$DEVICE batch-size=$BATCH_SIZE ! \
        gvawatermark ! \
        videoconvert ! \
        fpsdisplaysink video-sink=fakesink text-overlay=false
}

# Test different configurations
echo "Testing CPU performance..."
run_pipeline_docker "videos/test_video.mp4" "CPU" 1

echo "Testing GPU performance..."
run_pipeline_docker "videos/test_video.mp4" "GPU" 1

echo "Docker pipeline testing complete!"
EOF

chmod +x scripts/run_docker.sh
```

## Phase 6: Execution & Testing

### 6.1 Run the Complete Pipeline
```bash
# Make sure Docker is running
sudo systemctl start docker

# Run all tests
echo "Starting DL Streamer Pipeline Testing..."

# Test 1: Basic pipeline functionality
echo "1. Testing basic pipeline..."
docker run --rm --device /dev/dri -v $(pwd):/workspace -w /workspace intel/dlstreamer:latest python3 scripts/basic_detection.py

# Test 2: Full pipeline with classification
echo "2. Testing full pipeline..."
docker run --rm --device /dev/dri -v $(pwd):/workspace -w /workspace intel/dlstreamer:latest python3 scripts/full_pipeline.py

# Test 3: Multi-stream testing
echo "3. Testing multi-stream performance..."
docker run --rm --device /dev/dri -v $(pwd):/workspace -w /workspace intel/dlstreamer:latest python3 scripts/multi_stream_test.py

# Test 4: Performance analysis
echo "4. Analyzing performance..."
python3 scripts/analyze_performance.py

echo "All tests completed!"
```

### 6.2 Generate Final Report
```bash
cat > scripts/generate_report.py << 'EOF'
#!/usr/bin/env python3
import json
import subprocess
from datetime import datetime

def generate_final_report():
    """Generate the final 3-page report"""
    
    # Load all results
    try:
        with open("results/performance_summary.json", "r") as f:
            summary = json.load(f)
    except:
        summary = {"error": "Performance analysis not completed"}
    
    # Get system information
    try:
        cpu_info = subprocess.check_output(["lscpu"]).decode()
        gpu_info = subprocess.check_output(["lspci | grep VGA"], shell=True).decode()
    except:
        cpu_info = "CPU info not available"
        gpu_info = "GPU info not available"
    
    report = f"""
# DL Streamer Pipeline Performance Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Student:** [Your Name]
**Project:** Intel Unnati Industrial Training - Summer 2025

## Executive Summary

This report presents the performance analysis of a video analytics pipeline implemented using Intel DL Streamer framework. The pipeline incorporates object detection, video decoding, and classification capabilities optimized for Intel hardware.

### Key Findings:
- **CPU Performance:** {summary.get('scalability_summary', {}).get('best_cpu_performance', 'N/A')} FPS
- **GPU Performance:** {summary.get('scalability_summary', {}).get('best_gpu_performance', 'N/A')} FPS
- **Max CPU Streams:** {summary.get('multi_stream_summary', {}).get('max_cpu_streams', 'N/A')}
- **Max GPU Streams:** {summary.get('multi_stream_summary', {}).get('max_gpu_streams', 'N/A')}

## System Configuration

### Hardware Specifications:
```
{cpu_info}
```

### GPU Information:
```
{gpu_info}
```

## Pipeline Architecture

The implemented pipeline follows the architecture: **Detect → Decode → Classify**

### Components:
1. **Input Source:** filesrc (MP4 video files)
2. **Decoder:** decodebin with Intel QuickSync acceleration
3. **Detection:** gvadetect with person-detection-retail-0013 model
4. **Classification:** gvaclassify with age-gender-recognition-retail-0013 model
5. **Output:** gvawatermark for annotation

### GStreamer Pipeline:
```
gst-launch-1.0 filesrc location=input.mp4 ! decodebin ! videoconvert ! 
gvadetect model=person-detection-retail-0013.xml device=CPU batch-size=1 ! 
gvaclassify model=age-gender-recognition-retail-0013.xml device=CPU batch-size=1 ! 
gvawatermark ! videoconvert ! fpsdisplaysink
```

## Performance Analysis

### Single Stream Performance:
- **Optimal CPU Batch Size:** {summary.get('scalability_summary', {}).get('optimal_cpu_batch', 'N/A')}
- **Optimal GPU Batch Size:** {summary.get('scalability_summary', {}).get('optimal_gpu_batch', 'N/A')}

### Multi-Stream Scalability:
- **CPU Concurrency:** Up to {summary.get('multi_stream_summary', {}).get('max_cpu_streams', 'N/A')} concurrent streams
- **GPU Concurrency:** Up to {summary.get('multi_stream_summary', {}).get('max_gpu_streams', 'N/A')} concurrent streams

### Bottleneck Analysis:
Based on performance monitoring:
- **Primary Bottleneck:** [CPU/GPU/Memory/IO - to be determined from your results]
- **Resource Utilization:** [Details from your performance analysis]

## Recommendations

### For Production Deployment:
1. **CPU Optimization:** Use batch processing for better throughput
2. **GPU Acceleration:** Leverage Intel iGPU for real-time processing
3. **Memory Management:** Implement efficient buffer management
4. **Network Optimization:** Use UDP streaming for low latency

### Scalability Improvements:
1. **Horizontal Scaling:** Deploy multiple pipeline instances
2. **Load Balancing:** Distribute streams across available hardware
3. **Caching:** Implement model caching for faster initialization

## Conclusion

The Intel DL Streamer framework provides excellent performance for video analytics applications. The pipeline successfully demonstrates:
- Efficient object detection and classification
- Hardware acceleration capabilities
- Scalable multi-stream processing

The implementation meets the requirements for real-world deployment in smart city and transportation scenarios.

## References

1. Intel DL Streamer Documentation: https://dlstreamer.github.io/
2. OpenVINO Toolkit: https://docs.openvino.ai/
3. GStreamer Framework: https://gstreamer.freedesktop.org/
"""
    
    with open("reports/final_report.md", "w") as f:
        f.write(report)
    
    print("Final report generated: reports/final_report.md")

if __name__ == "__main__":
    generate_final_report()
EOF

chmod +x scripts/generate_report.py
```

## Phase 7: Final Execution Steps

### 7.1 Execute Complete Pipeline
```bash
# Run the complete project
./scripts/run_docker.sh
python3 scripts/generate_report.py

# Convert report to PDF (optional)
sudo apt install pandoc texlive-latex-base -y
pandoc reports/final_report.md -o reports/final_report.pdf
```

### 7.2 Project Deliverables
After completion, you will have:
- ✅ Working detection + classification pipeline
- ✅ Performance benchmarks for CPU and GPU
- ✅ Multi-stream scalability analysis
- ✅ Bottleneck identification  
- ✅ Performance visualization charts

## Expected Outcomes

Based on typical Intel hardware performance:
- **CPU:** 5-15 FPS with 2-4 concurrent streams
- **GPU:** 15-30 FPS with 4-8 concurrent streams
- **Primary Bottleneck:** Usually CPU for inference, GPU memory for high batch sizes
- **Optimal Configuration:** GPU with batch_size=2-4 for best performance

## Troubleshooting

### Common Issues:
1. **Docker permission denied:** `sudo usermod -aG docker $USER` and restart
2. **GPU not detected:** Install Intel GPU drivers and verify with `vainfo`
3. **Model download fails:** Use alternative model zoo or manual download
4. **Pipeline fails:** Check GStreamer plugins with `gst-inspect-1.0`

### Debug Commands:
```bash
# Check DL Streamer installation
docker run --rm intel/dlstreamer:latest gst-inspect-1.0 | grep gva

# Test GPU acceleration
docker run --rm --device /dev/dri intel/dlstreamer:latest vainfo

# Check available models
ls -la models/

# Test simple pipeline
docker run --rm -v $(pwd):/workspace -w /workspace intel/dlstreamer:latest \
gst-launch-1.0 videotestsrc num-buffers=100 ! video/x-raw,width=640,height=480 ! fakesink
```

## Additional Resources

### Official Documentation:
- [DL Streamer Developer Guide](https://dlstreamer.github.com/dev_guide/dev_guide_index.html)
- [Intel OpenVINO Documentation](https://docs.openvino.ai/)
- [GStreamer Documentation](https://gstreamer.freedesktop.org/documentation/)

### Sample Commands for Testing:
```bash
# Test with webcam input (if available)
docker run --rm --device /dev/video0 --device /dev/dri \
-v $(pwd):/workspace -w /workspace intel/dlstreamer:latest \
gst-launch-1.0 v4l2src device=/dev/video0 ! \
gvadetect model=models/person-detection-retail-0013.xml ! \
gvawatermark ! autovideosink

# Test with RTSP stream
docker run --rm --device /dev/dri -v $(pwd):/workspace -w /workspace \
intel/dlstreamer:latest gst-launch-1.0 \
rtspsrc location=rtsp://your-stream-url ! \
rtph264depay ! avdec_h264 ! \
gvadetect model=models/person-detection-2013.xml ! \
gvawatermark ! autovideosink
```

## Project Timeline

### Week 1: Setup and Basic Implementation
- Day 1-2: Environment setup and model download
- Day 3-4: Basic pipeline implementation
- Day 5-7: Testing and debugging

### Week 2: Performance Analysis
- Day 1-3: Multi-stream testing
- Day 4-5: Performance analysis and optimization
- Day 6-7: Report generation and documentation

## Expected Learning Outcomes

By completing this project, you will gain expertise in:
- **Video Analytics Pipeline Design**
- **Intel Hardware Optimization**
- **Real-time Stream Processing**
- **Performance Benchmarking**
- **System Scalability Analysis**

## Submission Checklist

- [ ] Complete pipeline implementation
- [ ] Performance benchmarks (CPU and GPU)
- [ ] Multi-stream scalability results
- [ ] Bottleneck analysis report
- [ ] 3-page technical report
- [ ] Source code with documentation
- [ ] Test results and charts  

## Additional Enhancements (Optional)

### Advanced Features:
1. **Custom Model Integration:** Train and deploy custom detection models
2. **Real-time Streaming:** Implement RTSP/WebRTC streaming
3. **Database Integration:** Store analytics results in database
4. **Web Dashboard:** Create web interface for monitoring
5. **Edge Deployment:** Deploy on Intel NUC or similar edge devices

### Code Example for Custom Model:
```python
# Custom model inference example
def load_custom_model(model_path):
    from openvino.inference_engine import IECore
    
    ie = IECore()
    model = ie.read_network(model_path)
    executable_network = ie.load_network(model, "CPU")
    
    return executable_network

# Integration with pipeline
def create_custom_pipeline(custom_model_path):
    pipeline = f"""
    gst-launch-1.0 filesrc location=input.mp4 ! 
    decodebin ! videoconvert ! 
    gvadetect model={custom_model_path} ! 
    gvawatermark ! videoconvert ! 
    autovideosink
    """
    return pipeline
```

## Performance Optimization Tips

### 1. Model Optimization:
- Use FP16 precision for faster inference
- Implement dynamic batch sizing
- Use model quantization for edge deployment

### 2. Pipeline Optimization:
- Minimize video conversions
- Use hardware-accelerated decode/encode
- Implement efficient buffer management

### 3. System Optimization:
- Set appropriate CPU affinity
- Use NUMA-aware memory allocation
- Optimize I/O operations

## Troubleshooting Common Issues

### Issue 1: "No such file or directory" for models
```bash
# Solution: Ensure models are downloaded
python3 scripts/download_models.py
ls -la models/  # Verify files exist
```

### Issue 2: GPU not being used
```bash
# Solution: Check GPU drivers and permissions
sudo apt install intel-media-va-driver-non-free
# Add user to video group
sudo usermod -a -G video $USER
```

### Issue 3: Poor performance
```bash
# Solution: Optimize pipeline parameters
# Try different batch sizes
# Use appropriate device (CPU/GPU)
# Check system resources with htop
```

### Issue 4: Memory issues
```bash
# Solution: Implement proper memory management
# Use smaller batch sizes
# Limit concurrent streams
# Monitor with: docker stats
```

## Final Notes

This project simulates real-world video analytics scenarios used in smart cities and transportation systems. The pipeline you build can be extended for:

- **Traffic monitoring** (vehicle detection and counting)
- **Security surveillance** (person detection and tracking)
- **Crowd management** (density estimation and flow analysis)
- **Infrastructure monitoring** (automated inspection systems)

The performance metrics you collect will help determine the optimal configuration for deploying such systems in production environments.

Good luck with your implementation!
