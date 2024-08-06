#pragma once
// ŞÜKRÜ ÇİRİŞ 2024

namespace SKR
{
    __global__ void getNegative(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, unsigned int count);

    __global__ void getLighter(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, unsigned char value, unsigned int count);

    __global__ void getDarker(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, unsigned char value, unsigned int count);

    __global__ void getLowContrast(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, int value, unsigned int count);

    __global__ void getHighContrast(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, int value, unsigned int count);

    __global__ void getSmooth(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, int width, int height);
};