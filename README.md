# QingMing Engine (青冥)

**High-Performance On-Device Vector Search for the Edge**  
**高性能端侧暴力向量搜索引擎**

---

### 📖 Introduction / 简介

**QingMing (青冥)** is a cross-platform, brute-force vector search engine engineered for the edge. By optimizing for memory bandwidth and cache residency, it achieves exact (or near-exact) recall with zero index construction time.

青冥是一个专为边缘计算打造的跨平台暴力向量搜索引擎。通过极致优化内存带宽和缓存驻留，它在无需构建索引的情况下，实现了全精度的检索效果。

**Supported Architectures / 支持架构:**
*   **NVIDIA GPU**: `qingming.cu` (Optimized for RTX 5090D / Server GPUs)
*   **AMD GPU**: `qingming.cpp` (HIP/ROCm for 7900 XTX)
*   **Mobile NPU/CPU**: `qingming-mobile.cpp` (NEON-accelerated for Snapdragon/Apple Silicon)

**Data Source / 数据来源:**
[ANN-Benchmarks Datasets](https://github.com/erikbern/ann-benchmarks?tab=readme-ov-file)

---

## 🚀 Quick Start / 快速启动

### 2.1 NVIDIA RTX 5090D v2 (24GB)
**Environment:** Ubuntu 24.04 + CUDA 12.8

#### Build / 编译
```bash
# Compiler Version: nvcc 12.8.93 (sm_120)
nvcc -O3 -arch=sm_120 qingming-flat.cu -o flat-hdf5 \
    -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial \
    -lhdf5_cpp -lhdf5
```

#### Benchmarks / 性能实测

**1. SIFT-1M (128-dim)**
> **Result:** 9354 QPS @ ~5.5ms Latency (Batch=10k) | Recall@1: 99.26%
```text
=======================================================
   QINGMING-ENGINE v1.0.0 PRO [GPU-FLAT]
   ADAPTER: NVIDIA RTX 5090D v2 (24GB)
=======================================================
Dataset Vectors : 1,000,000
Dimension       : 128
VRAM Usage      : 497.17 MB
-------------------------------------------------------
Recall@1        : 99.260 % (FP32 Precision)
Recall@10       : 100.000 %
-------------------------------------------------------
Max Throughput  : 9354.902 QPS
Latency P50     : 5.365 ms
Latency P99     : 5.573 ms
=======================================================
```

**2. GIST-1M (960-dim)**
> **Result:** High-Dimensional Throughput Test
```text
Dataset Vectors : 1,000,000
Dimension       : 960
VRAM Usage      : 3702.74 MB
-------------------------------------------------------
Recall@1        : 99.400 %
Recall@10       : 100.000 %
-------------------------------------------------------
Max Throughput  : 621.076 QPS
Latency P99     : 1.186 ms
=======================================================
```

**3. Deep-10M (96-dim)**
> **Result:** 10 Million Vectors Flat Search
```text
Dataset Vectors : 9,990,000
Dimension       : 96
VRAM Usage      : 3666.12 MB
-------------------------------------------------------
Recall@1        : 99.960 %
Recall@10       : 99.990 %
-------------------------------------------------------
Max Throughput  : 1137.779 QPS
Latency P99     : 1.188 ms
=======================================================
```

---

### 2.2 AMD Radeon RX 7900 XTX (24GB)
**Environment:** Ubuntu 24.04 + ROCm 6.2

#### Build / 编译
```bash
/opt/rocm-6.2.4/bin/amdclang++ -x hip -O3 --offload-arch=gfx1100 qingming.cpp -o qingming_amd \
    -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial \
    -lhdf5_cpp -lhdf5
```

#### Benchmarks / 性能实测

**SIFT-1M (128-dim)**
```text
=======================================================
QINGMING-ENGINE v1.0.0 PRO [REDMOON]
PLATFORM: AMD Radeon RX 7900 XTX (24GB)
Dataset Vectors : 1,000,000
Dimension       : 128
Recall@1        : 99.260 %
Max Throughput  : 6275.720 QPS
Latency P99     : 11.214 ms
=======================================================
```

**GIST-1M (960-dim)**
```text
Dataset Vectors : 1,000,000
Dimension       : 960
Recall@1        : 99.400 %
Max Throughput  : 470.285 QPS
Latency P99     : 25.755 ms
=======================================================
```

---

### 2.3 Mobile: Xiaomi 17 Pro Max (Snapdragon 8 Gen 5)
**Core Tech:** NEON SIMD + L3 Cache Residency (Zero-Copy)

#### Build / 编译
```bash
$TOOLCHAIN/aarch64-linux-android34-clang++ -O3 -static-libstdc++ -flto \
    -march=armv8.2-a+fp16+dotprod qingming-mobile.cpp -o qingming_8gen5
```

#### Deployment / 部署
```bash
adb push qingming_8gen5 /data/local/tmp/
adb shell chmod +x /data/local/tmp/qingming_8gen5
adb shell /data/local/tmp/qingming_8gen5 100000
```

#### Performance / 性能表现
> **Latency:** Brute-force search over **100k 128-dim vectors** achieves **~8ms** latency per query.  
> **Power:** 1000 consecutive queries resulted in **negligible thermal increase**.

---

## 💼 Licensing & Enterprise Support / 商业授权

QingMing Engine offers a dual-licensing model optimized for Edge and Server deployments.
青冥引擎提供针对边缘端和服务器端的双轨授权模式。

### 📱 QingMing-Mobile
**Target:** Android, iOS, Automotive (Smart Cockpit), IoT, Smart Cameras.  
**Tech:** ARM CPU + NEON Optimization.

| Region | Pricing Strategy | Terms |
| :--- | :--- | :--- |
| **Global** | **$9.99** / device / year | Converts to **perpetual license** after 3 years. <br>Or **$19.99** one-time perpetual. |
| **China** | **¥9.99** / 设备 / 年 | 3 年后自动转为永久授权。<br>或 **¥19.99** 一次性永久授权。 |

### 💻 QingMing-GPU
**Target:** NVIDIA / AMD Servers, Workstations.  
**Tech:** CUDA / HIP.

| Region | Pricing Strategy | Terms |
| :--- | :--- | :--- |
| **Global** | **$99** / GPU / year | Or **$199** one-time perpetual. |
| **China** | **¥99** / GPU / 年 | 或 **¥199** 一次性永久授权。 |

> **Note:** Vehicles/IoT devices using the CPU-based `qingming-mobile.cpp` fall under **Mobile Pricing**, even if a GPU is present.

---

### ✉️ Contact / 联系我们

**Email:** zhangxiaolong950@gmail.com

*Volume licensing, source code access, and custom enterprise integration available upon request.*  
*支持批量授权、源码交付与企业级定制方案。*
