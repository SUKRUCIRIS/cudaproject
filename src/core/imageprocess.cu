#include "imageprocess.h"
#include "utility.h"
// ŞÜKRÜ ÇİRİŞ 2024

__global__ void getNegative(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, unsigned int count)
{
    unsigned int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (tindex < count)
    {
        r_in[tindex] = 255 - r_in[tindex];
        g_in[tindex] = 255 - g_in[tindex];
        b_in[tindex] = 255 - b_in[tindex];
    }
}

__global__ void getLighter(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, unsigned char value, unsigned int count)
{
    unsigned int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (tindex < count)
    {
        r_in[tindex] = r_in[tindex] > (255 - value) ? 255 : r_in[tindex] + value;
        g_in[tindex] = g_in[tindex] > (255 - value) ? 255 : g_in[tindex] + value;
        b_in[tindex] = b_in[tindex] > (255 - value) ? 255 : b_in[tindex] + value;
    }
}

__global__ void getDarker(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, unsigned char value, unsigned int count)
{
    unsigned int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (tindex < count)
    {
        r_in[tindex] = r_in[tindex] > value ? r_in[tindex] - value : 0;
        g_in[tindex] = g_in[tindex] > value ? g_in[tindex] - value : 0;
        b_in[tindex] = b_in[tindex] > value ? b_in[tindex] - value : 0;
    }
}

__global__ void getLowContrast(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, int value, unsigned int count)
{
    unsigned int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (tindex < count)
    {
        r_in[tindex] = r_in[tindex] / value;
        g_in[tindex] = g_in[tindex] / value;
        b_in[tindex] = b_in[tindex] / value;
    }
}

__global__ void getHighContrast(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, int value, unsigned int count)
{
    unsigned int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (tindex < count)
    {
        r_in[tindex] = r_in[tindex] * value > 255 ? 255 : r_in[tindex] * value;
        g_in[tindex] = g_in[tindex] * value > 255 ? 255 : g_in[tindex] * value;
        b_in[tindex] = b_in[tindex] * value > 255 ? 255 : b_in[tindex] * value;
    }
}