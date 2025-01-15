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

        __global__ void getMultsOneSingle(float *data1, float data2, unsigned int count);

        __global__ void getNonSquaredDeviations(unsigned char *data, float mean, float *nsd, unsigned int count);

        __global__ void getSquaredDeviations(unsigned char *data, float mean, float *sd, unsigned int count);

        __global__ void getSobelEdges(float *sobelmag, float *minv, float *maxv, int width, int height, unsigned char threshold, unsigned char *out);

        __global__ void splitSingleChannel(unsigned char *in, unsigned char **out, int width, int height, int splitwidth, int splitheight);

        __global__ void splitSingleChannelTemplate(unsigned char *in, unsigned char **out, int width, int height, int splitwidth, int splitheight);

        __global__ void splitSingleChannelTemplateIndex(unsigned char *in, unsigned char *out, int width, int height, int splitwidth, int splitheight, int index);

        __global__ void splitSingleChannelTemplateIndexMultiple(unsigned char *in, unsigned char **out, int width, int height, int splitwidth, int splitheight, int index, int count);

        __global__ void getSSIM(float *mean1, float mean2, float *variance1, float variance2, float *covariance, unsigned int count, float *result, float c1, float c2);
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

        // pre_mean is optional, if you have already calculated mean, you can pass it to the functions to avoid recalculation

        float getVariance(Image *img, float *pre_mean = 0);

        float getStandardDeviation(Image *img, float *pre_mean = 0);

        float getCovariance(Image *img1, Image *img2, float *pre_mean1 = 0, float *pre_mean2 = 0);

        float getSSIM(Image *img1, Image *img2, float K1 = 0.01F, float K2 = 0.03F, float L = 255.0F,
                      float *pre_mean1 = 0, float *pre_mean2 = 0);

        float getSSIMOneIsPreCalculated(Image *img1, Image *img2, float pre_mean2, float pre_variance2,
                                        float K1 = 0.01F, float K2 = 0.03F, float L = 255.0F, float *pre_mean1 = 0);

        // I was gonna use this function for template matching but it uses so much GPU memory so I decided to not use it
        std::vector<Image *> *extractCandidatesForMatching(Image *img, int splitwidth, int splitheight);

        Image *extractCandidateForMatchingIndex(Image *img, int splitwidth, int splitheight, int index);

        std::vector<Image *> *extractCandidatesForMatchingIndexMultiple(Image *img, int splitwidth, int splitheight,
                                                                        int index, int count);

        // preallocated memory version of the function above so it is faster
        // img_out must be preallocated on CPU with count * sizeof(Image)
        // out must be preallocated on GPU with count * sizeof(unsigned char *)
        // h_out must be preallocated on CPU with count * sizeof(unsigned char *)
        // and each element of h_out must be preallocated on GPU with splitwidth * splitheight
        // h_out must be memcpyed to out before calling this function with cudaMemcpyHostToDevice
        // h_out and out are for internal use, only use img_out for further processing
        // after you are done with the processing of the batches, you must free the memories
        // if you want to continue to use images, you shouldn't free h_out elements, you should just free h_out itself
        // h_out elements are the image pixel pointers, if you free them, you will lose the images
        // when you free the images, h_out elements will be freed automatically
        void extractCandidatesForMatchingIndexMultiplePreAllocated(Image *img, int splitwidth, int splitheight,
                                                                   int index, int count, unsigned char **h_out,
                                                                   unsigned char **out, unsigned char **img_out);

        // preallocated memory version of the sum function so it is faster
        // sum1 and sum2 must be preallocated on CPU with batch_count * sizeof(float *)
        // sum1 and sum2 members must be preallocated on GPU with ((pixel_count + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK) * sizeof(float)
        // sum1 and sum2 are for internal use, only use result value for further processing
        // result must be preallocated on GPU with batch_count * sizeof(float)
        void getSumMultiplePreAllocated(unsigned char **img, unsigned int pixel_count,
                                        unsigned int batch_count, float **sum1, float **sum2, float *result);

        // preallocated memory version of the sum function so it is faster
        // sum1 and sum2 must be preallocated on CPU with batch_count * sizeof(float *)
        // sum1 and sum2 members must be preallocated on GPU with ((pixel_count + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK) * sizeof(float)
        // sum1 and sum2 are for internal use, only use result value for further processing
        // result must be preallocated on GPU with batch_count * sizeof(float)
        void getSumMultiplePreAllocated(float **img, unsigned int pixel_count, unsigned int batch_count,
                                        float **sum1, float **sum2, float *result);

        // preallocated memory version of the mean function so it is faster
        // sum1 and sum2 must be preallocated on CPU with batch_count * sizeof(float *)
        // sum1 and sum2 members must be preallocated on GPU with ((pixel_count + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK) * sizeof(float)
        // sum1, sum2 are for internal use, only use result value for further processing
        // result must be preallocated on GPU with batch_count * sizeof(float)
        void getMeanMultiplePreAllocated(unsigned char **img, unsigned int pixel_count,
                                         unsigned int batch_count, float **sum1,
                                         float **sum2, float *result);

        // preallocated memory version of the variance function so it is faster
        // sd must be preallocated on CPU with batch_count * sizeof(float *)
        // sd members must be preallocated on GPU with pixel_count * sizeof(float)
        // result must be preallocated on GPU with batch_count * sizeof(float)
        // sum1 and sum2 must be preallocated on CPU with batch_count * sizeof(float *)
        // sum1 and sum2 members must be preallocated on GPU with ((pixel_count + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK) * sizeof(float)
        void getVarianceMultiplePreAllocated(unsigned char **img, unsigned int pixel_count,
                                             unsigned int batch_count, float *pre_mean, float **sd,
                                             float **sum1, float **sum2, float *result);

        // preallocated memory version of the covariance function so it is faster
        // sd1 must be preallocated on CPU with batch_count * sizeof(float *)
        // sd1 members must be preallocated on GPU with pixel_count * sizeof(float)
        // sd2 must be preallocated on GPU with pixel_count * sizeof(float)
        // mults must be preallocated on CPU with batch_count * sizeof(float *)
        // mults members must be preallocated on GPU with pixel_count * sizeof(float)
        // result must be preallocated on GPU with batch_count * sizeof(float)
        // sum1 and sum2 must be preallocated on CPU with batch_count * sizeof(float *)
        // sum1 and sum2 members must be preallocated on GPU with ((pixel_count + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK) * sizeof(float)
        void getCovarianceMultiplePreAllocated(unsigned char **img1, unsigned char *img2, unsigned int pixel_count,
                                               unsigned int batch_count, float *pre_mean1, float pre_mean2, float **sd1,
                                               float *sd2, float **mults, float **sum1, float **sum2, float *result);

        // result must be preallocated on GPU with batch_count * sizeof(float)
        void getSSIMMultiplePreAllocated(float *pre_mean1, float pre_mean2, float *pre_variance1, float pre_variance2,
                                         float *covariance, unsigned int batch_count, float *result,
                                         float K1 = 0.01F, float K2 = 0.03F, float L = 255.0F);
    };
};