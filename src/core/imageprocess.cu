#include "imageprocess.h"
#include "utility.h"
// ŞÜKRÜ ÇİRİŞ 2024

__global__ void SKR::getNegative(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, unsigned int count)
{
    unsigned int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (tindex < count)
    {
        r_in[tindex] = 255 - r_in[tindex];
        g_in[tindex] = 255 - g_in[tindex];
        b_in[tindex] = 255 - b_in[tindex];
    }
}

__global__ void SKR::getLighter(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, unsigned char value, unsigned int count)
{
    unsigned int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (tindex < count)
    {
        r_in[tindex] = SET_UCHAR(r_in[tindex] + value);
        g_in[tindex] = SET_UCHAR(g_in[tindex] + value);
        b_in[tindex] = SET_UCHAR(b_in[tindex] + value);
    }
}

__global__ void SKR::getDarker(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, unsigned char value, unsigned int count)
{
    unsigned int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (tindex < count)
    {
        r_in[tindex] = SET_UCHAR(r_in[tindex] - value);
        g_in[tindex] = SET_UCHAR(g_in[tindex] - value);
        b_in[tindex] = SET_UCHAR(b_in[tindex] - value);
    }
}

__global__ void SKR::getLowContrast(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, int value, unsigned int count)
{
    unsigned int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (tindex < count)
    {
        r_in[tindex] = r_in[tindex] / value;
        g_in[tindex] = g_in[tindex] / value;
        b_in[tindex] = b_in[tindex] / value;
    }
}

__global__ void SKR::getHighContrast(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, int value, unsigned int count)
{
    unsigned int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (tindex < count)
    {
        r_in[tindex] = SET_UCHAR(r_in[tindex] * value);
        g_in[tindex] = SET_UCHAR(g_in[tindex] * value);
        b_in[tindex] = SET_UCHAR(b_in[tindex] * value);
    }
}

__device__ void runkernel5x5(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, int width, int height, float kernel[25], unsigned int tindex)
{
}

__global__ void SKR::getSmooth(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, int width, int height)
{
    unsigned int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (tindex < width * height)
    {
        float kernel[25] = {
            0.04f, 0.04f, 0.04f, 0.04f, 0.04f, //
            0.04f, 0.04f, 0.04f, 0.04f, 0.04f, //
            0.04f, 0.04f, 0.04f, 0.04f, 0.04f, //
            0.04f, 0.04f, 0.04f, 0.04f, 0.04f, //
            0.04f, 0.04f, 0.04f, 0.04f, 0.04f, //
        };
        runkernel5x5(r_in, g_in, b_in, width, height, kernel, tindex);
    }
}