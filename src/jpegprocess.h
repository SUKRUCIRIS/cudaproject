#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <nvjpeg.h>
#include <cuda_runtime.h>
// ŞÜKRÜ ÇİRİŞ 2024

struct jpegimage
{
    nvjpegImage_t image;
    int width;
    int height;
};

class jpegprocess
{
private:
    jpegprocess();
    ~jpegprocess();
    nvjpegHandle_t handle;
    nvjpegJpegState_t state;
    cudaStream_t stream;
    nvjpegEncoderState_t enc_state;
    nvjpegEncoderParams_t enc_params;

public:
    jpegprocess(const jpegprocess &) = delete;
    jpegprocess &operator=(const jpegprocess &) = delete;

    // singletone
    static jpegprocess &getInstance();

    // will return encoded jpeg data
    std::vector<unsigned char> *readJPEG(const std::string &filename);

    // will return planar rgb channels, width, height
    jpegimage *decodeJPEG(const std::vector<unsigned char> &jpeg_buffer);

    // will return encoded jpeg data, quality must be between 1 and 100
    std::vector<unsigned char> *encodeJPEG(const jpegimage *image, const int quality, const bool isgray);

    // write encoded jpeg data to a file
    void writeJPEG(const std::string &filename, const std::vector<unsigned char> &jpeg_buffer);

    // frees cuda memory and deletes the nvjpegImage_t structure
    void freeJPEG(jpegimage *image);
};