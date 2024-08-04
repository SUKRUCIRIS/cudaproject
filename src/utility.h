#pragma once
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