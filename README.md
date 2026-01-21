# 1 简介
## 青冥：高性能端侧向量搜索引擎  
**Qingming: High-Performance On-Device Vector Search**

青冥是一个为边缘计算打造的跨平台暴力向量搜索引擎。  
Qingming is a cross-platform, brute-force vector search engine built for the edge.

- qingming.cu NVIDIA RTX5090D v2 24G
- qingming.cpp AMD 7900 XTX 24G
- qingming-mobile.cpp Xiaomi 17 Pro Max
# 2 快速启动 quickly start
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
### 2.2.1 
## 2.3 Xiaomi 17 Pro Max 
