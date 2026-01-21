/*
==================================================================================
 * QINGMING-ENGINE: High-Performance GPU Vector Search Kernel
 * Copyright (C) 2024–2026 Xiaolong Zhang. All rights reserved.
 * ==================================================================================
 *
 * LICENSE:
 * Free for personal use, academic research, and organizations with ≤10 employees.
 * Commercial use in larger organizations or SaaS products requires a license.
 * See full terms at: https://github.com/uulong950/qingming-flat/LICENSE
 *
 * NO WARRANTY — USE AT YOUR OWN RISK.
 * ==================================================================================
 * PROJECT: QINGMING-ENGINE v1.0.0 PRO
 * PLATFORM: NVIDIA RTX 5090D V2 (24GB)
 * ARCHITECTURE: HYBRID (Warp-Scan for Throughput / Map-Reduce for Latency)
 * RECALL: 100% (Exact Search)
 * ==================================================================================
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <numeric>
#include <H5Cpp.h>

using namespace H5;

#define TOP_K 100
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
#define BLOCKS_FOR_LATENCY 256 

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

// =================================================================================
// KERNEL STRATEGY 1: THROUGHPUT (One Warp Per Query)
// Best for Batch > 100
// =================================================================================
template <int DIM>
__global__ void kernel_throughput_scan(
    const float* __restrict__ queries,       
    const float* __restrict__ database,      
    int n_base,                              
    int* __restrict__ out_ids,               
    int batch_size                           
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int qid = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id;
    
    if (qid >= batch_size) return;

    constexpr int ITER = DIM / WARP_SIZE;
    float my_q[ITER];
    const float* q_ptr = queries + qid * DIM;
    
    #pragma unroll
    for (int i = 0; i < ITER; ++i) {
        my_q[i] = q_ptr[lane_id + i * WARP_SIZE];
    }

    float best_dists[TOP_K];
    int best_ids[TOP_K];
    
    if (lane_id == 0) {
        for(int k=0; k<TOP_K; ++k) { best_dists[k] = 1e30f; best_ids[k] = -1; }
    }

    for (int i = 0; i < n_base; ++i) {
        float dist = 0.0f;
        const float* vec_ptr = database + i * DIM;
        
        #pragma unroll
        for (int k = 0; k < ITER; ++k) {
            float v = vec_ptr[lane_id + k * WARP_SIZE];
            float diff = my_q[k] - v;
            dist += diff * diff;
        }

        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) 
            dist += __shfl_down_sync(FULL_MASK, dist, offset);
        
        if (lane_id == 0) {
            if (dist < best_dists[TOP_K-1]) {
                int pos = TOP_K - 1;
                while (pos > 0 && best_dists[pos-1] > dist) {
                    best_dists[pos] = best_dists[pos-1];
                    best_ids[pos] = best_ids[pos-1];
                    pos--;
                }
                best_dists[pos] = dist;
                best_ids[pos] = i; 
            }
        }
    }

    if (lane_id == 0) {
        for (int k = 0; k < TOP_K; ++k) {
            out_ids[qid * TOP_K + k] = best_ids[k];
        }
    }
}

// =================================================================================
// KERNEL STRATEGY 2: LATENCY (Map-Reduce)
// Best for Batch = 1
// =================================================================================

// Step A: Map (Partial Scan)
template <int DIM>
__global__ void kernel_latency_map(
    const float* __restrict__ query,         
    const float* __restrict__ database,      
    int n_base,                              
    int* __restrict__ partial_ids,           
    float* __restrict__ partial_dists        
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int num_blocks = gridDim.x;

    int chunk_size = (n_base + num_blocks - 1) / num_blocks;
    int start_idx = bid * chunk_size;
    int end_idx = min(start_idx + chunk_size, n_base);

    __shared__ float s_q[DIM];
    for (int i = tid; i < DIM; i += blockDim.x) s_q[i] = query[i];
    __syncthreads();

    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    float best_dists[TOP_K];
    int best_ids[TOP_K];
    for(int k=0; k<TOP_K; ++k) { best_dists[k] = 1e30f; best_ids[k] = -1; }

    constexpr int ITER = DIM / WARP_SIZE;

    for (int i = start_idx + warp_id; i < end_idx; i += blockDim.x / WARP_SIZE) {
        float dist = 0.0f;
        const float* vec_ptr = database + i * DIM;

        #pragma unroll
        for (int k = 0; k < ITER; ++k) {
            float v = vec_ptr[lane_id + k * WARP_SIZE];
            float diff = s_q[lane_id + k * WARP_SIZE] - v;
            dist += diff * diff;
        }

        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) 
            dist += __shfl_down_sync(FULL_MASK, dist, offset);

        if (lane_id == 0) {
            if (dist < best_dists[TOP_K-1]) {
                int pos = TOP_K - 1;
                while (pos > 0 && best_dists[pos-1] > dist) {
                    best_dists[pos] = best_dists[pos-1];
                    best_ids[pos] = best_ids[pos-1];
                    pos--;
                }
                best_dists[pos] = dist;
                best_ids[pos] = i; 
            }
        }
    }

    __shared__ float s_block_dists[4 * TOP_K]; 
    __shared__ int s_block_ids[4 * TOP_K];

    if (lane_id == 0) {
        for(int k=0; k<TOP_K; ++k) {
            s_block_dists[warp_id * TOP_K + k] = best_dists[k];
            s_block_ids[warp_id * TOP_K + k] = best_ids[k];
        }
    }
    __syncthreads();

    if (warp_id == 0 && lane_id == 0) {
        for(int k=0; k<TOP_K; ++k) { best_dists[k] = 1e30f; best_ids[k] = -1; }

        for (int w = 0; w < blockDim.x / WARP_SIZE; ++w) {
            for (int k = 0; k < TOP_K; ++k) {
                float d = s_block_dists[w * TOP_K + k];
                int id = s_block_ids[w * TOP_K + k];
                if (id == -1) break; 

                if (d < best_dists[TOP_K-1]) {
                    int pos = TOP_K - 1;
                    while (pos > 0 && best_dists[pos-1] > d) {
                        best_dists[pos] = best_dists[pos-1];
                        best_ids[pos] = best_ids[pos-1];
                        pos--;
                    }
                    best_dists[pos] = d;
                    best_ids[pos] = id;
                }
            }
        }

        for(int k=0; k<TOP_K; ++k) {
            partial_dists[bid * TOP_K + k] = best_dists[k];
            partial_ids[bid * TOP_K + k] = best_ids[k];
        }
    }
}

// Step B: Reduce (Final Merge)
__global__ void kernel_latency_reduce(
    const int* __restrict__ partial_ids,
    const float* __restrict__ partial_dists,
    int num_partials, 
    int* __restrict__ final_ids
) {
    if (threadIdx.x != 0) return;

    float best_dists[TOP_K];
    int best_ids[TOP_K];
    for(int k=0; k<TOP_K; ++k) { best_dists[k] = 1e30f; best_ids[k] = -1; }

    for (int i = 0; i < num_partials; ++i) {
        float d = partial_dists[i];
        int id = partial_ids[i];
        if (id == -1) continue;

        if (d < best_dists[TOP_K-1]) {
            int pos = TOP_K - 1;
            while (pos > 0 && best_dists[pos-1] > d) {
                best_dists[pos] = best_dists[pos-1];
                best_ids[pos] = best_ids[pos-1];
                pos--;
            }
            best_dists[pos] = d;
            best_ids[pos] = id;
        }
    }

    for (int k = 0; k < TOP_K; ++k) {
        final_ids[k] = best_ids[k];
    }
}

// =================================================================================
// DATA LOADER
// =================================================================================
class HDF5Loader {
public:
    static void load(const std::string& path, std::vector<float>& train, std::vector<float>& test, std::vector<int>& gt, int& dim) {
        try {
            H5File file(path, H5F_ACC_RDONLY);
            auto load_f = [&](const std::string& name, std::vector<float>& vec) {
                DataSet ds = file.openDataSet(name);
                hsize_t dims[2]; ds.getSpace().getSimpleExtentDims(dims, NULL);
                vec.resize(dims[0] * dims[1]); ds.read(vec.data(), PredType::NATIVE_FLOAT);
                if (name == "train") dim = dims[1];
            };
            auto load_i = [&](const std::string& name, std::vector<int>& vec) {
                DataSet ds = file.openDataSet(name);
                hsize_t dims[2]; ds.getSpace().getSimpleExtentDims(dims, NULL);
                vec.resize(dims[0] * dims[1]); ds.read(vec.data(), PredType::NATIVE_INT);
            };
            load_f("train", train); load_f("test", test); load_i("neighbors", gt);
        } catch (...) { 
            std::cerr << "Fatal Error: Failed to load HDF5 file." << std::endl; 
            exit(1); 
        }
    }
};

// =================================================================================
// QINGMING ENGINE
// =================================================================================
class QingMingEngine {
    float *d_database;
    float *d_queries;
    int *d_results;
    
    // Latency buffers
    float *d_query_single;
    int *d_part_ids;
    float *d_part_dists;
    int *d_final_ids;

    int dimension;
    int n_base;
    size_t vram_usage;

public:
    QingMingEngine(const std::vector<float>& db_data, int dim) : dimension(dim) {
        n_base = db_data.size() / dim;
        size_t db_size = db_data.size() * sizeof(float);
        
        // 1. Main Database
        CUDA_CHECK(cudaMalloc(&d_database, db_size));
        CUDA_CHECK(cudaMemcpy(d_database, db_data.data(), db_size, cudaMemcpyHostToDevice));
        
        // 2. Throughput Buffers (Max 10000 queries)
        size_t max_q = 10000;
        CUDA_CHECK(cudaMalloc(&d_queries, max_q * dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_results, max_q * TOP_K * sizeof(int)));

        // 3. Latency Buffers
        CUDA_CHECK(cudaMalloc(&d_query_single, dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_final_ids, TOP_K * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_part_ids, BLOCKS_FOR_LATENCY * TOP_K * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_part_dists, BLOCKS_FOR_LATENCY * TOP_K * sizeof(float)));

        vram_usage = db_size + (max_q * dim * 4) + (max_q * TOP_K * 4) + (BLOCKS_FOR_LATENCY * TOP_K * 8);
    }

    // Strategy 1: High Throughput (Batch Saturation)
    void search_saturation(const std::vector<float>& queries, std::vector<int>& results) {
        int n_q = queries.size() / dimension;
        results.resize(n_q * TOP_K);
        CUDA_CHECK(cudaMemcpy(d_queries, queries.data(), queries.size() * 4, cudaMemcpyHostToDevice));

        int warps_per_block = 4;
        int threads = warps_per_block * WARP_SIZE;
        int blocks = (n_q + warps_per_block - 1) / warps_per_block;

        if (dimension == 128) kernel_throughput_scan<128><<<blocks, threads>>>(d_queries, d_database, n_base, d_results, n_q);
        else if (dimension == 96) kernel_throughput_scan<96><<<blocks, threads>>>(d_queries, d_database, n_base, d_results, n_q);
        else if (dimension == 960) kernel_throughput_scan<960><<<blocks, threads>>>(d_queries, d_database, n_base, d_results, n_q);
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(results.data(), d_results, results.size() * 4, cudaMemcpyDeviceToHost));
    }

    // Strategy 2: Low Latency (Single Query Parallel)
    void search_latency(const std::vector<float>& query, std::vector<int>& result) {
        CUDA_CHECK(cudaMemcpy(d_query_single, query.data(), dimension * 4, cudaMemcpyHostToDevice));

        if (dimension == 128) 
            kernel_latency_map<128><<<BLOCKS_FOR_LATENCY, 128>>>(d_query_single, d_database, n_base, d_part_ids, d_part_dists);
        
        kernel_latency_reduce<<<1, 1>>>(d_part_ids, d_part_dists, BLOCKS_FOR_LATENCY * TOP_K, d_final_ids);

        result.resize(TOP_K);
        CUDA_CHECK(cudaMemcpy(result.data(), d_final_ids, TOP_K * 4, cudaMemcpyDeviceToHost));
    }

    size_t get_vram() { return vram_usage; }
};

int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "sift-128-euclidean.hdf5";
    std::vector<float> train, test;
    std::vector<int> gt;
    int dim = 0;

    std::cout << "[INFO] Loading HDF5 Dataset..." << std::endl;
    HDF5Loader::load(path, train, test, gt, dim);
    int n_base = train.size() / dim;
    int n_q = test.size() / dim;

    QingMingEngine engine(train, dim);
    std::vector<int> res_sat;

    // 1. Warmup
    std::cout << "[INFO] Warming up GPU pipeline..." << std::endl;
    std::vector<float> warm_q(test.begin(), test.begin() + 100*dim);
    engine.search_saturation(warm_q, res_sat);

    // 2. Throughput Test
    std::cout << "[BENCH] Running Saturation Test (Batch=" << n_q << ")..." << std::endl;
    auto t1_start = std::chrono::high_resolution_clock::now();
    engine.search_saturation(test, res_sat);
    auto t1_end = std::chrono::high_resolution_clock::now();
    
    double sat_ms = std::chrono::duration<double, std::milli>(t1_end - t1_start).count();
    double sat_qps = n_q * 1000.0 / sat_ms;

    // 3. Latency Test
    std::cout << "[BENCH] Running Latency Test (Batch=1, Samples=1000)..." << std::endl;
    std::vector<double> latencies;
    int samples = 1000;
    std::vector<int> res_lat;

    for(int i=0; i<samples; ++i) {
        std::vector<float> q_one(test.begin() + i*dim, test.begin() + (i+1)*dim);
        auto start = std::chrono::high_resolution_clock::now();
        engine.search_latency(q_one, res_lat);
        auto end = std::chrono::high_resolution_clock::now();
        latencies.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(latencies.begin(), latencies.end());

    // 4. Recall Verification
    int k1=0, k10=0;
    for(int i=0; i<n_q; ++i) {
        int truth = gt[i * 100];
        for(int k=0; k<TOP_K; ++k) {
            if (res_sat[i*TOP_K + k] == truth) {
                if(k==0) k1++; if(k<10) k10++; break;
            }
        }
    }

    // FINAL REPORT
    std::cout << "\n=======================================================\n";
    std::cout << "   QINGMING-ENGINE v1.0.0 PRO [GPU-FLAT]\n";
    std::cout << "   ADAPTER: NVIDIA RTX 5090D v2 (24GB)\n";
    std::cout << "=======================================================\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Dataset Vectors : " << n_base << "\n";
    std::cout << "Dimension       : " << dim << "\n";
    std::cout << "VRAM Usage      : " << engine.get_vram() / 1024.0 / 1024.0 << " MB\n";
    std::cout << "-------------------------------------------------------\n";
    std::cout << "Recall@1        : " << 100.0 * k1 / n_q << " %\n";
    std::cout << "Recall@10       : " << 100.0 * k10 / n_q << " %\n";
    std::cout << "-------------------------------------------------------\n";
    std::cout << "Max Throughput  : " << sat_qps << " QPS\n";
    std::cout << "Latency P50     : " << latencies[(int)(samples*0.50)] << " ms\n";
    std::cout << "Latency P95     : " << latencies[(int)(samples*0.95)] << " ms\n";
    std::cout << "Latency P99     : " << latencies[(int)(samples*0.99)] << " ms\n";
    std::cout << "Latency P999    : " << latencies[(int)(samples*0.999)] << " ms\n";
    std::cout << "=======================================================\n";

    return 0;
}
