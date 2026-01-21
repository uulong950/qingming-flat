# 1 简介
## 青冥：高性能端侧向量搜索引擎  
**Qingming: High-Performance On-Device Vector Search**

青冥是一个为边缘计算打造的跨平台暴力向量搜索引擎。  
Qingming is a cross-platform, brute-force vector search engine built for the edge.

- qingming.cu NVIDIA RTX5090D v2 24G
- qingming.cpp AMD 7900 XTX 24G
- qingming-mobile.cpp Xiaomi 17 Pro Max
# 2 快速启动 quickly start
数据来源 Data Source
https://github.com/erikbern/ann-benchmarks?tab=readme-ov-file
## 2.1 NVIDIA RTX5090D v2 24G + Ubuntu24.04 + cuda
### 2.1.1 cuda
```bash
nvcc: NVIDIA (R) Cuda compiler driver 
Copyright (c) 2005-2025 NVIDIA Corporation 
Built on Fri_Feb_21_20:23:50_PST_2025 
Cuda compilation tools, release 12.8, V12.8.93 
Build cuda_12.8.r12.8/compiler.35583870_0 
```
### 2.1.2 compile
```bash
nvcc -O3 -arch=sm_120 qingming-flat.cu -o flat-hdf5 -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5_cpp -lhdf5

./flat-hdf5 ./data/sift-128-euclidean.hdf5
./flat-hdf5 ./data/gist-960-euclidean.hdf5
./flat-hdf5 ./data/deep-image-96-angular.hdf5
```
### 2.1.3 result
#### 2.1.3.1 SIFT-1M 128
```bash
[INFO] Loading HDF5 Dataset...
[INFO] Warming up GPU pipeline...
[BENCH] Running Saturation Test (Batch=10000)...
[BENCH] Running Latency Test (Batch=1, Samples=1000)...

=======================================================
   QINGMING-ENGINE v1.0.0 PRO [GPU-FLAT]
   ADAPTER: NVIDIA RTX 5090D v2 (24GB)
=======================================================
Dataset Vectors : 1000000
Dimension       : 128
VRAM Usage      : 497.174 MB
-------------------------------------------------------
Recall@1        : 99.260 %
Recall@10       : 100.000 %
-------------------------------------------------------
Max Throughput  : 9354.902 QPS
Latency P50     : 5.365 ms
Latency P95     : 5.501 ms
Latency P99     : 5.573 ms
Latency P999    : 5.598 ms
=======================================================
```
#### 2.1.3.2 GIST-1M 960
```bash
[INFO] Loading HDF5 Dataset...
[INFO] Warming up GPU pipeline...
[BENCH] Running Saturation Test (Batch=1000)...
[BENCH] Running Latency Test (Batch=1, Samples=1000)...

=======================================================
   QINGMING-ENGINE v1.0.0 PRO [GPU-FLAT]
   ADAPTER: NVIDIA RTX 5090D v2 (24GB)
=======================================================
Dataset Vectors : 1000000
Dimension       : 960
VRAM Usage      : 3702.740 MB
-------------------------------------------------------
Recall@1        : 99.400 %
Recall@10       : 100.000 %
-------------------------------------------------------
Max Throughput  : 621.076 QPS
Latency P50     : 1.177 ms
Latency P95     : 1.179 ms
Latency P99     : 1.186 ms
Latency P999    : 1.858 ms
=======================================================
```
#### 2.1.3.3 Deep-10M 96
```bash
[INFO] Loading HDF5 Dataset...
[INFO] Warming up GPU pipeline...
[BENCH] Running Saturation Test (Batch=10000)...
[BENCH] Running Latency Test (Batch=1, Samples=1000)...

=======================================================
   QINGMING-ENGINE v1.0.0 PRO [GPU-FLAT]
   ADAPTER: NVIDIA RTX 5090D v2 (24GB)
=======================================================
Dataset Vectors : 9990000
Dimension       : 96
VRAM Usage      : 3666.119 MB
-------------------------------------------------------
Recall@1        : 99.960 %
Recall@10       : 99.990 %
-------------------------------------------------------
Max Throughput  : 1137.779 QPS
Latency P50     : 1.185 ms
Latency P95     : 1.186 ms
Latency P99     : 1.188 ms
Latency P999    : 1.875 ms
=======================================================
```
## 2.2 AMD 7900 XTX 24G + Ubuntu24.04 + ROCM 6.2
### 2.2.1 rocm
```bash
/opt/rocm-6.2.4/bin/amdclang++
```
### 2.2.2 compile
```bash
/opt/rocm-6.2.4/bin/amdclang++ -x hip -O3 --offload-arch=gfx1100 qingming.cpp -o qingming_amd -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5_cpp -lhdf5
```
### 2.2.3 result
#### 2.2.3.1 SIFT-1M 128 
```bash
[BENCH] Running Saturation Test (Batch=10000)...
[BENCH] Running Latency Test (Graph + GPU Reduce)...
[BENCH] Validating Recall...
=======================================================
QINGMING-ENGINE v1.0.0 PRO [REDMOON]
PLATFORM: AMD Radeon RX 7900 XTX (24GB)
Dataset Vectors : 1000000
Dimension       : 128
VRAM Usage      : 493.237 MB
Recall@1        : 99.260 %
Recall@10       : 100.000 %
Max Throughput  : 6275.720 QPS
Latency P50     : 10.572 ms
Latency P95     : 11.000 ms
Latency P99     : 11.214 ms
Latency P999    : 11.500 ms
```
#### 2.2.3.2 GIST-10M 96
```bash
[BENCH] Running Saturation Test (Batch=1000)...
[BENCH] Launching Persistent Agent...
[BENCH] Running Latency Test (Persistent Mode)...
[BENCH] Validating Recall...
=======================================================
QINGMING-ENGINE v1.0.0 PRO [REDMOON]
PLATFORM: AMD Radeon RX 7900 XTX (24GB)
Dataset Vectors : 1000000
Dimension       : 960
VRAM Usage      : 3698.804 MB
Recall@1        : 99.400 %
Recall@10       : 100.000 %
Max Throughput  : 470.285 QPS
Latency P50     : 20.087 ms
Latency P95     : 23.714 ms
Latency P99     : 25.755 ms
Latency P999    : 27.237 ms
```
## 2.3 Xiaomi 17 Pro Max 
...
## 💼 Commercial Licensing / 商业授权

### 📱 Qingming-Mobile（适用于 Android、iOS、车机、IoT）  
*Runs on ARM CPUs with NEON (e.g., Snapdragon, Kirin, MediaTek)*  
*支持 ARM CPU + NEON 指令集（如高通骁龙、华为麒麟、联发科等）*

- **中国区**：¥9.99 / 设备 / 年 → **3 年后自动转为永久授权**  
  或 ¥19.99 一次性永久授权  
- **国际区**：$9.99 / device / year → **converts to perpetual license after 3 years**  
  or $19.99 one-time perpetual license  

✅ 适用设备包括：智能手机、智能座舱（车机）、IoT 设备、智能摄像头等  
✅ Includes: smartphones, automotive infotainment systems, IoT devices, smart cameras, etc.

---

### 💻 Qingming-GPU（适用于 NVIDIA / AMD 服务器）  
*Requires CUDA or HIP-compatible GPU*  
*需 CUDA 或 HIP 兼容的 GPU*

- **中国区**：¥99 / GPU / 年，或 ¥199 一次性永久授权  
- **国际区**：$99 / GPU / year, or $199 one-time perpetual license  

---

> 🔹 **Mobile 与 GPU 为两个独立产品线。**  
> 🔹 **Mobile and GPU are separate products.**  
>   
> 若车辆使用的是 `qingming-mobile.cpp`（基于 CPU/NEON），则按 **Mobile 定价**，**不适用 GPU 授权费用**。  
> A vehicle using `qingming-mobile.cpp` (CPU/NEON-based) is licensed under **Mobile pricing** — **GPU pricing does NOT apply**.

---

### ✉️ 联系我们 / Contact  
邮箱 / Email: zhangxiaolong950@gmail.com  
支持批量授权与企业定制方案 / Volume licensing and enterprise agreements available.
