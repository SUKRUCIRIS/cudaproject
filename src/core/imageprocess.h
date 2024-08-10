#pragma once
#include "jpegprocess.h"
// ŞÜKRÜ ÇİRİŞ 2024

namespace SKR
{
    // user shouldn't access this namespace
    namespace kernels
    {
        __global__ void getNegative(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, unsigned int count);

        __global__ void getLighter(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, unsigned char value, unsigned int count);

        __global__ void getDarker(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, unsigned char value, unsigned int count);

        __global__ void getLowContrast(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, int value, unsigned int count);

        __global__ void getHighContrast(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, int value, unsigned int count);

        __global__ void getSmooth(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, int width, int height,
                                  unsigned char *r_out, unsigned char *g_out, unsigned char *b_out);
    };

    class imageprocess
    {
    private:
        cudaStream_t stream;
        imageprocess();
        ~imageprocess();

    public:
        imageprocess(const imageprocess &) = delete;
        imageprocess &operator=(const imageprocess &) = delete;

        // singletone
        static imageprocess &getInstance();

        // every pixel will be substracted from 255
        void getNegative(jpegimage *img);

        // value will be added to every pixel
        void getLighter(jpegimage *img, unsigned char value);

        // value will be substracted from every pixel
        void getDarker(jpegimage *img, unsigned char value);

        // value will be divided by every pixel
        void getLowContrast(jpegimage *img, int value);

        // value will be muliplied by every pixel
        void getHighContrast(jpegimage *img, int value);

        // image will be smoothed by a 5x5 kernel
        void getSmooth(jpegimage *img);
    };
};