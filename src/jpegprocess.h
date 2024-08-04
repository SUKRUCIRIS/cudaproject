#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <nvjpeg.h>

class jpegprocess
{
private:
    jpegprocess();
    ~jpegprocess();
    nvjpegHandle_t handle;
    nvjpegJpegState_t state;
    cudaStream_t stream;

public:
    jpegprocess(const jpegprocess &) = delete;
    jpegprocess &operator=(const jpegprocess &) = delete;

    static jpegprocess& getInstance();

    bool loadJPEG(const std::string &filename, std::vector<unsigned char> &buffer);

    void decodeJPEG(const std::vector<unsigned char> &jpeg_buffer, std::vector<std::vector<unsigned char>> &output_buffers);
};