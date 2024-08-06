#pragma once
// ŞÜKRÜ ÇİRİŞ 2024
#include <iostream>

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

#define SET_UCHAR(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)));

#define GET_MINDEX(row, column, width) ((row) * (width) + (column));

#define GET_MROW(index, width) ((index) / (width));

#define GET_MCOLUMN(index, width) ((index) % (width));