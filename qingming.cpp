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
 * PLATFORM: AMD RX 7900 XTX (24GB)
 * ARCHITECTURE: HYBRID (Warp-Scan for Throughput / Map-Reduce for Latency)
 * RECALL: 100% (Exact Search)
 * ==================================================================================
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <numeric>
#include <H5Cpp.h>
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
#include <stdlib.h> 

using namespace H5;

#define TOP_K 100
#define WAVE_SIZE 32           
#define THREADS_PER_BLOCK 256  
#define BLOCKS_FOR_LATENCY 96 

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP Error: %s at line %d\n", hipGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

__device__ inline float wave_reduce_sum(float val) {
    for (int offset = WAVE_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// =================================================================================
// KERNELS
// =================================================================================

// Throughput (Float4 - SIFT128)
template <int DIM>
__global__ __launch_bounds__(256) 
void kernel_throughput_float4(const float4* __restrict__ queries, const float4* __restrict__ database, int n_base, int* __restrict__ out_ids, int batch_size) {
    int wave_id = threadIdx.x / WAVE_SIZE; int lane_id = threadIdx.x % WAVE_SIZE;
    int qid = blockIdx.x * (blockDim.x / WAVE_SIZE) + wave_id;
    if (qid >= batch_size) return;
    constexpr int ITER = (DIM / 4) / WAVE_SIZE; 
    float4 my_q[ITER];
    const float4* q_ptr = queries + qid * (DIM / 4);
    #pragma unroll
    for(int i=0; i<ITER; ++i) my_q[i] = q_ptr[lane_id + i*WAVE_SIZE];
    float best_dists[TOP_K]; int best_ids[TOP_K];
    if (lane_id == 0) for(int k=0; k<TOP_K; ++k) { best_dists[k] = 1e30f; best_ids[k] = -1; }
    for (int i = 0; i < n_base; ++i) {
        const float4* vec_ptr = database + i * (DIM / 4);
        float dist = 0.0f;
        #pragma unroll
        for(int k=0; k<ITER; ++k) {
            float4 v = vec_ptr[lane_id + k*WAVE_SIZE];
            dist += (my_q[k].x - v.x)*(my_q[k].x - v.x) + (my_q[k].y - v.y)*(my_q[k].y - v.y) + (my_q[k].z - v.z)*(my_q[k].z - v.z) + (my_q[k].w - v.w)*(my_q[k].w - v.w);
        }
        dist = wave_reduce_sum(dist);
        if (lane_id == 0 && dist < best_dists[TOP_K-1]) {
            int pos = TOP_K - 1; while (pos > 0 && best_dists[pos-1] > dist) { best_dists[pos] = best_dists[pos-1]; best_ids[pos] = best_ids[pos-1]; pos--; }
            best_dists[pos] = dist; best_ids[pos] = i; 
        }
    }
    if (lane_id == 0) for (int k = 0; k < TOP_K; ++k) out_ids[qid * TOP_K + k] = best_ids[k];
}

// Throughput (Float - GIST960)
template <int DIM>
__global__ __launch_bounds__(256) 
void kernel_throughput_float(const float* __restrict__ queries, const float* __restrict__ database, int n_base, int* __restrict__ out_ids, int batch_size) {
    int wave_id = threadIdx.x / WAVE_SIZE; int lane_id = threadIdx.x % WAVE_SIZE;
    int qid = blockIdx.x * (blockDim.x / WAVE_SIZE) + wave_id;
    if (qid >= batch_size) return;
    constexpr int ITER = DIM / WAVE_SIZE; 
    float my_q[ITER];
    const float* q_ptr = queries + qid * DIM;
    #pragma unroll
    for(int i=0; i<ITER; ++i) my_q[i] = q_ptr[lane_id + i*WAVE_SIZE];
    float best_dists[TOP_K]; int best_ids[TOP_K];
    if (lane_id == 0) for(int k=0; k<TOP_K; ++k) { best_dists[k] = 1e30f; best_ids[k] = -1; }
    for (int i = 0; i < n_base; ++i) {
        const float* vec_ptr = database + i * DIM;
        float dist = 0.0f;
        #pragma unroll
        for(int k=0; k<ITER; ++k) {
            float v = vec_ptr[lane_id + k*WAVE_SIZE]; float diff = my_q[k] - v; dist += diff * diff;
        }
        dist = wave_reduce_sum(dist);
        if (lane_id == 0 && dist < best_dists[TOP_K-1]) {
            int pos = TOP_K - 1; while (pos > 0 && best_dists[pos-1] > dist) { best_dists[pos] = best_dists[pos-1]; best_ids[pos] = best_ids[pos-1]; pos--; }
            best_dists[pos] = dist; best_ids[pos] = i; 
        }
    }
    if (lane_id == 0) for (int k = 0; k < TOP_K; ++k) out_ids[qid * TOP_K + k] = best_ids[k];
}

// Latency Map (Float4)
template <int DIM>
__global__ __launch_bounds__(256)
void kernel_latency_map_float4(const float4* __restrict__ query, const float4* __restrict__ database, int n_base, int* __restrict__ partial_ids, float* __restrict__ partial_dists) {
    int tid = threadIdx.x; int bid = blockIdx.x;
    int lane_id = tid % WAVE_SIZE; int wave_id = tid / WAVE_SIZE;
    constexpr int ITER = (DIM / 4) / WAVE_SIZE;
    float4 my_q[ITER];
    #pragma unroll
    for(int i=0; i<ITER; ++i) my_q[i] = query[lane_id + i*WAVE_SIZE];
    int chunk_size = (n_base + gridDim.x - 1) / gridDim.x;
    int start_idx = bid * chunk_size; int end_idx = min(start_idx + chunk_size, n_base);
    float best_dists[TOP_K]; int best_ids[TOP_K];
    for(int k=0; k<TOP_K; ++k) { best_dists[k] = 1e30f; best_ids[k] = -1; }
    int waves_in_block = blockDim.x / WAVE_SIZE;
    for (int i = start_idx + wave_id; i < end_idx; i += waves_in_block) {
        const float4* vec_ptr = database + i * (DIM / 4);
        float dist = 0.0f;
        #pragma unroll
        for(int k=0; k<ITER; ++k) {
            float4 v = vec_ptr[lane_id + k*WAVE_SIZE];
            dist += (my_q[k].x - v.x)*(my_q[k].x - v.x) + (my_q[k].y - v.y)*(my_q[k].y - v.y) + (my_q[k].z - v.z)*(my_q[k].z - v.z) + (my_q[k].w - v.w)*(my_q[k].w - v.w);
        }
        dist = wave_reduce_sum(dist);
        if (lane_id == 0 && dist < best_dists[TOP_K-1]) {
            int pos = TOP_K - 1; while (pos > 0 && best_dists[pos-1] > dist) { best_dists[pos] = best_dists[pos-1]; best_ids[pos] = best_ids[pos-1]; pos--; }
            best_dists[pos] = dist; best_ids[pos] = i; 
        }
    }
    // Block Reduce
    __shared__ float s_d[8 * TOP_K]; __shared__ int s_i[8 * TOP_K];
    if (lane_id == 0) {
        #pragma unroll
        for(int k=0; k<TOP_K; ++k) { s_d[wave_id*TOP_K+k] = best_dists[k]; s_i[wave_id*TOP_K+k] = best_ids[k]; }
    }
    __syncthreads();
    if (tid == 0) {
        for(int k=0; k<TOP_K; ++k) { best_dists[k] = 1e30f; best_ids[k] = -1; }
        for(int w=0; w<waves_in_block; ++w) {
            for(int k=0; k<TOP_K; ++k) {
                float d = s_d[w*TOP_K+k]; int id = s_i[w*TOP_K+k];
                if(id==-1) break;
                if(d < best_dists[TOP_K-1]) {
                    int pos = TOP_K-1; while(pos>0 && best_dists[pos-1]>d) { best_dists[pos]=best_dists[pos-1]; best_ids[pos]=best_ids[pos-1]; pos--; }
                    best_dists[pos]=d; best_ids[pos]=id;
                }
            }
        }
        int out_idx = bid * TOP_K;
        #pragma unroll
        for(int k=0; k<TOP_K; ++k) { partial_dists[out_idx+k] = best_dists[k]; partial_ids[out_idx+k] = best_ids[k]; }
    }
}

// Latency Map (Float)
template <int DIM>
__global__ __launch_bounds__(256)
void kernel_latency_map_float(const float* __restrict__ query, const float* __restrict__ database, int n_base, int* __restrict__ partial_ids, float* __restrict__ partial_dists) {
    int tid = threadIdx.x; int bid = blockIdx.x;
    int lane_id = tid % WAVE_SIZE; int wave_id = tid / WAVE_SIZE;
    constexpr int ITER = DIM / WAVE_SIZE;
    float my_q[ITER];
    #pragma unroll
    for(int i=0; i<ITER; ++i) my_q[i] = query[lane_id + i*WAVE_SIZE];
    int chunk_size = (n_base + gridDim.x - 1) / gridDim.x;
    int start_idx = bid * chunk_size; int end_idx = min(start_idx + chunk_size, n_base);
    float best_dists[TOP_K]; int best_ids[TOP_K];
    for(int k=0; k<TOP_K; ++k) { best_dists[k] = 1e30f; best_ids[k] = -1; }
    int waves_in_block = blockDim.x / WAVE_SIZE;
    for (int i = start_idx + wave_id; i < end_idx; i += waves_in_block) {
        const float* vec_ptr = database + i * DIM;
        float dist = 0.0f;
        #pragma unroll
        for(int k=0; k<ITER; ++k) {
            float v = vec_ptr[lane_id + k*WAVE_SIZE]; float diff = my_q[k] - v; dist += diff * diff;
        }
        dist = wave_reduce_sum(dist);
        if (lane_id == 0 && dist < best_dists[TOP_K-1]) {
            int pos = TOP_K - 1; while (pos > 0 && best_dists[pos-1] > dist) { best_dists[pos] = best_dists[pos-1]; best_ids[pos] = best_ids[pos-1]; pos--; }
            best_dists[pos] = dist; best_ids[pos] = i; 
        }
    }
    // Block Reduce
    __shared__ float s_d[8 * TOP_K]; __shared__ int s_i[8 * TOP_K];
    if (lane_id == 0) {
        #pragma unroll
        for(int k=0; k<TOP_K; ++k) { s_d[wave_id*TOP_K+k] = best_dists[k]; s_i[wave_id*TOP_K+k] = best_ids[k]; }
    }
    __syncthreads();
    if (tid == 0) {
        for(int k=0; k<TOP_K; ++k) { best_dists[k] = 1e30f; best_ids[k] = -1; }
        for(int w=0; w<waves_in_block; ++w) {
            for(int k=0; k<TOP_K; ++k) {
                float d = s_d[w*TOP_K+k]; int id = s_i[w*TOP_K+k];
                if(id==-1) break;
                if(d < best_dists[TOP_K-1]) {
                    int pos = TOP_K-1; while(pos>0 && best_dists[pos-1]>d) { best_dists[pos]=best_dists[pos-1]; best_ids[pos]=best_ids[pos-1]; pos--; }
                    best_dists[pos]=d; best_ids[pos]=id;
                }
            }
        }
        int out_idx = bid * TOP_K;
        #pragma unroll
        for(int k=0; k<TOP_K; ++k) { partial_dists[out_idx+k] = best_dists[k]; partial_ids[out_idx+k] = best_ids[k]; }
    }
}

// Reduce Kernel
__global__ void kernel_reduce_final(
    const int* __restrict__ partial_ids, const float* __restrict__ partial_dists,
    int num_partials, int* __restrict__ final_ids) 
{
    if (threadIdx.x != 0) return;
    float best_dists[TOP_K]; int best_ids[TOP_K];
    for(int k=0; k<TOP_K; ++k) { best_dists[k] = 1e30f; best_ids[k] = -1; }
    for (int i = 0; i < num_partials; ++i) {
        float d = partial_dists[i]; int id = partial_ids[i];
        if (id == -1) continue;
        if (d < best_dists[TOP_K-1]) {
            int pos = TOP_K - 1; while (pos > 0 && best_dists[pos-1] > d) { best_dists[pos] = best_dists[pos-1]; best_ids[pos] = best_ids[pos-1]; pos--; }
            best_dists[pos] = d; best_ids[pos] = id;
        }
    }
    for (int k = 0; k < TOP_K; ++k) final_ids[k] = best_ids[k];
}

// =================================================================================
// ENGINE
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
        } catch (...) { exit(1); }
    }
};

class QingMingEngine {
    float4 *d_database, *d_queries, *d_query_single;
    float *d_part_dists;
    int *d_results, *d_part_ids, *d_final_ids;
    
    // HIP Graph
    hipGraph_t graph;
    hipGraphExec_t graphExec;
    hipStream_t stream;
    bool graph_created = false;

    int dimension, n_base;
    size_t vram_usage;
    size_t part_count;

public:
    QingMingEngine(const std::vector<float>& db_data, int dim) : dimension(dim) {
        n_base = db_data.size() / dim;
        size_t db_size = db_data.size() * sizeof(float);
        
        HIP_CHECK(hipMalloc(&d_database, db_size));
        HIP_CHECK(hipMemcpy(d_database, db_data.data(), db_size, hipMemcpyHostToDevice));
        
        size_t max_q = 10000;
        HIP_CHECK(hipMalloc(&d_queries, max_q * dim * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_results, max_q * TOP_K * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_query_single, dim * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_final_ids, TOP_K * sizeof(int)));
        
        part_count = BLOCKS_FOR_LATENCY * TOP_K;
        HIP_CHECK(hipMalloc(&d_part_ids, part_count * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_part_dists, part_count * sizeof(float)));
        
        vram_usage = db_size + (max_q * dim * 4) + (part_count * 8);

        HIP_CHECK(hipSetDeviceFlags(hipDeviceScheduleSpin));
        HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, -1)); 
    }

    ~QingMingEngine() {
        if(graph_created) {
            (void)hipGraphExecDestroy(graphExec);
            (void)hipGraphDestroy(graph);
        }
        (void)hipStreamDestroy(stream);
        (void)hipFree(d_database); // ...
    }

    void build_graph() {
        if (graph_created) return;

        HIP_CHECK(hipGraphCreate(&graph, 0));

        // Node 1: Map Kernel
        void* kernel_func_map = NULL;
        if (dimension == 128) kernel_func_map = (void*)kernel_latency_map_float4<128>;
        else if (dimension == 960) kernel_func_map = (void*)kernel_latency_map_float<960>;

        hipKernelNodeParams kParamsMap = {0};
        kParamsMap.func = kernel_func_map;
        kParamsMap.gridDim = dim3(BLOCKS_FOR_LATENCY);
        kParamsMap.blockDim = dim3(THREADS_PER_BLOCK);
        kParamsMap.sharedMemBytes = 0;
        void* argsMap[] = {&d_query_single, &d_database, &n_base, &d_part_ids, &d_part_dists};
        kParamsMap.kernelParams = argsMap;
        kParamsMap.extra = NULL;
        
        hipGraphNode_t kNodeMap;
        HIP_CHECK(hipGraphAddKernelNode(&kNodeMap, graph, NULL, 0, &kParamsMap));

        // Node 2: Reduce Kernel (Depends on Map)
        hipKernelNodeParams kParamsRed = {0};
        kParamsRed.func = (void*)kernel_reduce_final;
        kParamsRed.gridDim = dim3(1);
        kParamsRed.blockDim = dim3(1);
        kParamsRed.sharedMemBytes = 0;
        int num_partials = part_count;
        void* argsRed[] = {&d_part_ids, &d_part_dists, &num_partials, &d_final_ids};
        kParamsRed.kernelParams = argsRed;
        kParamsRed.extra = NULL;

        hipGraphNode_t kNodeRed;
        HIP_CHECK(hipGraphAddKernelNode(&kNodeRed, graph, &kNodeMap, 1, &kParamsRed));

        HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
        graph_created = true;
    }

    void search_saturation(const std::vector<float>& queries, std::vector<int>& results) {
        int n_q = queries.size() / dimension;
        results.resize(n_q * TOP_K);
        HIP_CHECK(hipMemcpy(d_queries, queries.data(), queries.size() * 4, hipMemcpyHostToDevice));
        int threads = THREADS_PER_BLOCK; 
        int blocks = (n_q * WAVE_SIZE + threads - 1) / threads;

        if (dimension == 128) kernel_throughput_float4<128><<<blocks, threads>>>(d_queries, d_database, n_base, d_results, n_q);
        // [SABO FIX] Explicit cast to float* for non-float4 kernels
        else if (dimension == 960) kernel_throughput_float<960><<<blocks, threads>>>((float*)d_queries, (float*)d_database, n_base, d_results, n_q);
        
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(results.data(), d_results, results.size() * 4, hipMemcpyDeviceToHost));
    }

    void search_latency(const std::vector<float>& query, std::vector<int>& result) {
        HIP_CHECK(hipMemcpyAsync(d_query_single, query.data(), dimension * 4, hipMemcpyHostToDevice, stream));
        
        if (!graph_created) build_graph();
        HIP_CHECK(hipGraphLaunch(graphExec, stream));

        result.resize(TOP_K);
        HIP_CHECK(hipMemcpyAsync(result.data(), d_final_ids, TOP_K * 4, hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
    }
    size_t get_vram() { return vram_usage; }
};

int main(int argc, char** argv) {
    setenv("HSA_POLL_PUT", "1", 1);
    setenv("HSA_ENABLE_INTERRUPT", "0", 1);
    
    std::string path = (argc > 1) ? argv[1] : "sift-128-euclidean.hdf5";
    std::vector<float> train, test;
    std::vector<int> gt;
    int dim = 0;
    HDF5Loader::load(path, train, test, gt, dim);
    int n_base = train.size() / dim;
    int n_q = test.size() / dim;

    QingMingEngine engine(train, dim);
    std::vector<int> res_sat;

    // 1. Warmup
    std::vector<float> warm_q(test.begin(), test.begin() + 100*dim);
    engine.search_saturation(warm_q, res_sat);

    // 2. Saturation Test
    std::cout << "[BENCH] Running Saturation Test (Batch=" << n_q << ")..." << std::endl;
    auto t1_start = std::chrono::high_resolution_clock::now();
    engine.search_saturation(test, res_sat);
    auto t1_end = std::chrono::high_resolution_clock::now();
    double sat_ms = std::chrono::duration<double, std::milli>(t1_end - t1_start).count();
    double sat_qps = n_q * 1000.0 / sat_ms;

    // 3. Latency Test
    std::cout << "[BENCH] Running Latency Test (Graph + GPU Reduce)..." << std::endl;
    std::vector<double> latencies;
    int samples = 1000;
    std::vector<int> res_lat;
    
    engine.search_latency(warm_q, res_lat); // Graph Warmup

    auto spin_start = std::chrono::high_resolution_clock::now();
    std::vector<int> dummy;
    while(true) {
        engine.search_saturation(warm_q, dummy);
        if (std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - spin_start).count() > 1.0) break;
    }

    for(int i=0; i<samples; ++i) {
        std::vector<float> q_one(test.begin() + i*dim, test.begin() + (i+1)*dim);
        auto start = std::chrono::high_resolution_clock::now();
        engine.search_latency(q_one, res_lat);
        auto end = std::chrono::high_resolution_clock::now();
        latencies.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(latencies.begin(), latencies.end());

    // 4. Recall
    std::cout << "[BENCH] Validating Recall..." << std::endl;
    int k1=0, k10=0;
    for(int i=0; i<n_q; ++i) {
        int truth = gt[i * 100];
        for(int k=0; k<TOP_K; ++k) {
            if (res_sat[i*TOP_K + k] == truth) {
                if(k==0) k1++; if(k<10) k10++; break;
            }
        }
    }

    std::cout << "\n=======================================================\n";
    std::cout << "   QINGMING-ENGINE v1.0.0 PRO [REDMOON]\n";
    std::cout << "   PLATFORM: AMD Radeon RX 7900 XTX (24GB)\n";
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
