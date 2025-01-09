#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <nvjpeg.h>
#include <cuda_runtime.h>
// ŞÜKRÜ ÇİRİŞ 2024

namespace SKR
{

    struct Image
    {
        nvjpegImage_t image;
        int width;
        int height;
    };

    class jpegde
    {
    private:
        jpegde();
        ~jpegde();
        nvjpegHandle_t handle;
        nvjpegJpegState_t state;
        cudaStream_t stream;
        nvjpegEncoderState_t enc_state;
        nvjpegEncoderParams_t enc_params;

    public:
        jpegde(const jpegde &) = delete;
        jpegde &operator=(const jpegde &) = delete;

        // singletone
        static jpegde &getInstance();

        // will return encoded jpeg data
        std::vector<unsigned char> *readJPEG(const std::string &filename);

        // will return planar rgb channels which are allocated on GPU, width, height
        Image *decodeJPEG(const std::vector<unsigned char> &jpeg_buffer);

        // r, g, b channels must be allocated on GPU
        Image *createImage(unsigned char *r, unsigned char *g, unsigned char *b, int width, int height);

        // single channel must be allocated on GPU
        Image *createImage(unsigned char *r, int width, int height);

        // will return encoded jpeg data, quality must be between 1 and 100
        std::vector<unsigned char> *encodeJPEG(const Image *image, const int quality, const bool isgray);

        // write encoded jpeg data to a file
        void writeJPEG(const std::string &filename, const std::vector<unsigned char> &jpeg_buffer);

        // frees cuda memory and deletes the jpegimage structure
        void freeJPEG(Image *image);
    };
};