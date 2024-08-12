#pragma once
// ŞÜKRÜ ÇİRİŞ 2024
#include <iostream>
#include <chrono>

#define CHECK_CUDA(call)                                                                                          \
    {                                                                                                             \
        cudaError_t _e = (call);                                                                                  \
        if (_e != cudaSuccess)                                                                                    \
        {                                                                                                         \
            std::cout << "CUDA Runtime failure: '#" << _e << "' at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(-1);                                                                                             \
        }                                                                                                         \
    }

#define CHECK_NVJPEG(call)                                                                                  \
    {                                                                                                       \
        nvjpegStatus_t _e = (call);                                                                         \
        if (_e != NVJPEG_STATUS_SUCCESS)                                                                    \
        {                                                                                                   \
            std::cout << "NVJPEG failure: '#" << _e << "' at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(-1);                                                                                       \
        }                                                                                                   \
    }

#define SET_UCHAR(value) ((value) >= 255 ? 255 : ((value) <= 0 ? 0 : (unsigned char)(value)))

#define GET_MINDEX(row, column, width) ((row) * (width) + (column))

#define GET_MROW(index, width) ((index) / (width))

#define GET_MCOLUMN(index, width) ((index) % (width))

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define MIN(a, b) ((a) > (b) ? (b) : (a))

#define DIFF(a, b) (MAX((a), (b)) - MIN((a), (b)))

#define MIRROR(a, center, low, high) ((center) + (a)) < (low) ? ((center) - (a)) : (((center) + (a)) > (high) ? ((center) - (a)) : ((center) + (a)))

#define NORM(v, minv, maxv) (((v) - (minv)) / ((maxv) - (minv)))

#if (DEBUG)

#define MEASURE_TIME1 auto start_time = std::chrono::high_resolution_clock::now();

#define MEASURE_TIME2(name)                                                                        \
    auto stop_time = std::chrono::high_resolution_clock::now();                                    \
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time); \
    std::cout << (name) << " took " << duration.count() << " microseconds" << std::endl;

#else

#define MEASURE_TIME1
#define MEASURE_TIME2(name)

#endif