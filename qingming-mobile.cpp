/*
 * ==================================================================================
 * QINGMING MOBILE: ENDURANCE TEST V2 (Anti-Optimization)
 * SCENARIO: 10k Clicks, 30ms Interval
 * FIX: Force compiler to execute the loop via volatile checksum
 * ==================================================================================
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <arm_neon.h>
#include <unistd.h>
#include <cstdlib>
#include <thread>

#define DIMENSION 128
#define TOP_K 100
#define TOTAL_CLICKS 10000
#define CLICK_INTERVAL_MS 30 

void* aligned_malloc(size_t size) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 16, size) != 0) return nullptr;
    return ptr;
}

inline float l2_dist_neon(const float* __restrict__ q, const float* __restrict__ p) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    for (int i = 0; i < 128; i += 16) {
        float32x4_t d0 = vsubq_f32(vld1q_f32(q + i), vld1q_f32(p + i));
        sum = vmlaq_f32(sum, d0, d0);
        float32x4_t d1 = vsubq_f32(vld1q_f32(q + i + 4), vld1q_f32(p + i + 4));
        sum = vmlaq_f32(sum, d1, d1);
        float32x4_t d2 = vsubq_f32(vld1q_f32(q + i + 8), vld1q_f32(p + i + 8));
        sum = vmlaq_f32(sum, d2, d2);
        float32x4_t d3 = vsubq_f32(vld1q_f32(q + i + 12), vld1q_f32(p + i + 12));
        sum = vmlaq_f32(sum, d3, d3);
    }
    return vaddvq_f32(sum);
}

class QingMingEngineMobile {
    float* database;
    int n_base;
public:
    QingMingEngineMobile(int n) : n_base(n) {
        size_t bytes = n * DIMENSION * sizeof(float);
        database = (float*)aligned_malloc(bytes);
        // Fill random data
        for(size_t i=0; i<n*DIMENSION; ++i) {
             database[i] = (float)rand() / (float)RAND_MAX;
        }
    }
    
    ~QingMingEngineMobile() {
        free(database);
    }
    
    // Returns latency in NANOSECONDS
    long search(const float* query, int& out_id) {
        auto start = std::chrono::high_resolution_clock::now();
        
        float best_d = 1e30f;
        int best_id = -1;
        
        // Compiler Barrier to force memory read
        asm volatile("" ::: "memory");

        for(int i=0; i<n_base; ++i) {
            float d = l2_dist_neon(query, database + i * DIMENSION);
            if(d < best_d) { best_d = d; best_id = i; }
        }
        
        out_id = best_id;
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
};

int main(int argc, char** argv) {
    int N = 100000; 
    if(argc > 1) N = atoi(argv[1]);

    std::cout << "========================================" << std::endl;
    std::cout << "   QINGMING MOBILE: ENDURANCE TEST V2   " << std::endl;
    std::cout << "   Simulating " << TOTAL_CLICKS << " interactions (30ms interval)" << std::endl;
    std::cout << "========================================" << std::endl;

    QingMingEngineMobile engine(N);
    float query[DIMENSION];
    for(int i=0; i<DIMENSION; ++i) query[i] = (float)rand() / (float)RAND_MAX;

    int dummy = 0;
    engine.search(query, dummy); // Warmup

    long max_ns = 0;
    long min_ns = 999999999;
    long total_ns = 0;
    
    // [SABO FIX] Volatile accumulator to prevent optimization
    volatile long long checksum = 0;

    auto test_start = std::chrono::high_resolution_clock::now();

    for(int i=0; i<TOTAL_CLICKS; ++i) {
        // 1. Search
        int result_id = 0;
        long lat_ns = engine.search(query, result_id);
        
        // [SABO FIX] Force usage of result
        checksum += result_id;

        // 2. Stats
        total_ns += lat_ns;
        if(lat_ns > max_ns) max_ns = lat_ns;
        if(lat_ns < min_ns) min_ns = lat_ns;

        // 3. Race to Sleep
        // Latency is in nanoseconds. Sleep is in microseconds.
        // Sleep time = 30ms - latency
        long lat_us = lat_ns / 1000;
        long sleep_us = (CLICK_INTERVAL_MS * 1000) - lat_us;
        
        if (sleep_us > 0) {
            usleep(sleep_us);
        }

        if ((i+1) % 1000 == 0) {
            std::cout << "Completed " << (i+1) << "/" << TOTAL_CLICKS 
                      << " | Last: " << lat_ns / 1000000.0 << " ms" << std::endl;
        }
    }

    auto test_end = std::chrono::high_resolution_clock::now();
    double total_wall_time = std::chrono::duration<double>(test_end - test_start).count();

    std::cout << "========================================" << std::endl;
    std::cout << "TEST DURATION : " << total_wall_time << " s" << std::endl;
    std::cout << "CHECKSUM      : " << checksum << " (Verified)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "AVG LATENCY   : " << (total_ns / TOTAL_CLICKS) / 1000000.0 << " ms" << std::endl;
    std::cout << "MIN LATENCY   : " << min_ns / 1000000.0 << " ms" << std::endl;
    std::cout << "MAX LATENCY   : " << max_ns / 1000000.0 << " ms" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
