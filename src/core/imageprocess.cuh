#pragma once
#include "jpegde.cuh"
// ŞÜKRÜ ÇİRİŞ 2024

namespace SKR
{
    constexpr int MAX_CUDA_THREADS_PER_BLOCK = 1024; // Nvidia GPUs can have 256, 512 or 1024 thread per block

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

        __global__ void getGray(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, unsigned int count);

        __global__ void getSobel(unsigned char *in, int width, int height, float *sobelmag);

        __global__ void getMax(float *data, float *maxv, unsigned int count);

        __global__ void getMin(float *data, float *minv, unsigned int count);

        __global__ void getSum(unsigned char *data, float *sum, unsigned int count);

        __global__ void getSumFloat(float *data, float *sum, unsigned int count);

        __global__ void getMults(float *data1, float *data2, float *mults, unsigned int count);

        __global__ void getNonSquaredDeviations(unsigned char *data, float mean, float *nsd, unsigned int count);

        __global__ void getSquaredDeviations(unsigned char *data, float mean, float *sd, unsigned int count);

        __global__ void getSobelEdges(float *sobelmag, float *minv, float *maxv, int width, int height, unsigned char threshold, unsigned char *out);

        __global__ void splitSingleChannel(unsigned char *in, unsigned char **out, int width, int height, int splitwidth, int splitheight);
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
        void getNegative(Image *img);

        // value will be added to every pixel
        void getLighter(Image *img, unsigned char value);

        // value will be substracted from every pixel
        void getDarker(Image *img, unsigned char value);

        // value will be divided by every pixel
        void getLowContrast(Image *img, int value);

        // value will be muliplied by every pixel
        void getHighContrast(Image *img, int value);

        // image will be smoothed by a 5x5 kernel
        void getSmooth(Image *img);

        // image will be grayscale
        void getGray(Image *img);

        // finds edges in image. image must be grayscale
        void getSobelEdges(Image *img, unsigned char threshold);

        // get sum of the all pixel values
        float getSum(Image *img);

        // get sum of the all unsigned char values, unsigned char array must be on device
        float getSum(unsigned char *img, unsigned int count);

        // get sum of the all float values, float array must be on device
        float getSum(float *img, unsigned int count);

        // get minimum of all values, float array must be on device
        float getMin(float *img, unsigned int count);

        // get maximum of all values, float array must be on device
        float getMax(float *img, unsigned int count);

        // get mean of all pixel values
        float getMean(Image *img);

        // split single channel image to multiple images
        std::vector<Image *> *splitSingleChannel(Image *img, int splitwidth, int splitheight);

        float getVariance(Image *img);

        float getStandardDeviation(Image *img);

        float getCovariance(Image *img1, Image *img2);

        float getSSIM(Image *img1, Image *img2, float K1 = 0.01F, float K2 = 0.03F, float L = 255.0F);
    };
};