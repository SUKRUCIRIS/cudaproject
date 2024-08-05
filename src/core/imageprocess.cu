#include "imageprocess.h"
#include "utility.h"
// ŞÜKRÜ ÇİRİŞ 2024

__global__ void getNegative(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, int count)
{
    int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (tindex < count)
    {
        r_in[tindex] = 255 - r_in[tindex];
        g_in[tindex] = 255 - g_in[tindex];
        b_in[tindex] = 255 - b_in[tindex];
    }
}