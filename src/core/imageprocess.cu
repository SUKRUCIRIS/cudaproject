#include "imageprocess.cuh"
#include "utility.cuh"
// ŞÜKRÜ ÇİRİŞ 2024

__global__ void SKR::kernels::getNegative(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, unsigned int count)
{
    unsigned int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (tindex < count)
    {
        r_in[tindex] = 255 - r_in[tindex];
        g_in[tindex] = 255 - g_in[tindex];
        b_in[tindex] = 255 - b_in[tindex];
    }
}

__global__ void SKR::kernels::getLighter(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, unsigned char value, unsigned int count)
{
    unsigned int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (tindex < count)
    {
        r_in[tindex] = SET_UCHAR(r_in[tindex] + value);
        g_in[tindex] = SET_UCHAR(g_in[tindex] + value);
        b_in[tindex] = SET_UCHAR(b_in[tindex] + value);
    }
}

__global__ void SKR::kernels::getDarker(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, unsigned char value, unsigned int count)
{
    unsigned int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (tindex < count)
    {
        r_in[tindex] = SET_UCHAR(r_in[tindex] - value);
        g_in[tindex] = SET_UCHAR(g_in[tindex] - value);
        b_in[tindex] = SET_UCHAR(b_in[tindex] - value);
    }
}

__global__ void SKR::kernels::getLowContrast(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, int value, unsigned int count)
{
    unsigned int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (tindex < count)
    {
        r_in[tindex] = r_in[tindex] / value;
        g_in[tindex] = g_in[tindex] / value;
        b_in[tindex] = b_in[tindex] / value;
    }
}

__global__ void SKR::kernels::getHighContrast(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, int value, unsigned int count)
{
    unsigned int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (tindex < count)
    {
        r_in[tindex] = SET_UCHAR(r_in[tindex] * value);
        g_in[tindex] = SET_UCHAR(g_in[tindex] * value);
        b_in[tindex] = SET_UCHAR(b_in[tindex] * value);
    }
}

__global__ void SKR::kernels::getGray(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, unsigned int count)
{
    unsigned int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (tindex < count)
    {
        float gray = 0.299f * r_in[tindex] + 0.587f * g_in[tindex] + 0.114f * b_in[tindex];
        unsigned char grayu = SET_UCHAR(gray);
        r_in[tindex] = grayu;
        g_in[tindex] = grayu;
        b_in[tindex] = grayu;
    }
}

__device__ void runkernel5x5(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, int width, int height, float kernel[25], unsigned int tindex,
                             unsigned char *r_out, unsigned char *g_out, unsigned char *b_out)
{
    unsigned int column = GET_MCOLUMN(tindex, width);
    unsigned int row = GET_MROW(tindex, width);
    unsigned int maxcolumn = width - 1, maxrow = height - 1;
    unsigned int kernelcenter = 2;
    unsigned int kernelcolumn = 0, kernelrow = 0, mindex = 0;
    int coldiff = 0, rowdiff = 0;
    float sumr = 0, sumg = 0, sumb = 0;
    for (int i = 0; i < 25; i++)
    {
        kernelcolumn = GET_MCOLUMN(i, 5);
        kernelrow = GET_MROW(i, 5);
        coldiff = kernelcolumn - kernelcenter;
        rowdiff = kernelrow - kernelcenter;
        mindex = GET_MINDEX(MIRROR(rowdiff, row, 0U, maxrow), MIRROR(coldiff, column, 0U, maxcolumn), width);
        sumr += (r_in[mindex] * kernel[i]);
        sumg += (g_in[mindex] * kernel[i]);
        sumb += (b_in[mindex] * kernel[i]);
    }
    r_out[tindex] = SET_UCHAR(sumr);
    g_out[tindex] = SET_UCHAR(sumg);
    b_out[tindex] = SET_UCHAR(sumb);
}

__device__ void runkernel5x5Sobel(unsigned char *in, float *out, int width, int height, float kernel[25], unsigned int tindex)
{
    unsigned int column = GET_MCOLUMN(tindex, width);
    unsigned int row = GET_MROW(tindex, width);
    unsigned int maxcolumn = width - 1, maxrow = height - 1;
    unsigned int kernelcenter = 2;
    unsigned int kernelcolumn = 0, kernelrow = 0, mindex = 0;
    int coldiff = 0, rowdiff = 0;
    (*out) = 0;
    for (int i = 0; i < 25; i++)
    {
        kernelcolumn = GET_MCOLUMN(i, 5);
        kernelrow = GET_MROW(i, 5);
        coldiff = kernelcolumn - kernelcenter;
        rowdiff = kernelrow - kernelcenter;
        mindex = GET_MINDEX(MIRROR(rowdiff, row, 0U, maxrow), MIRROR(coldiff, column, 0U, maxcolumn), width);
        (*out) += (in[mindex] * kernel[i]);
    }
}

__constant__ float smooth_kernel[25] = {
    0.04f, 0.04f, 0.04f, 0.04f, 0.04f, //
    0.04f, 0.04f, 0.04f, 0.04f, 0.04f, //
    0.04f, 0.04f, 0.04f, 0.04f, 0.04f, //
    0.04f, 0.04f, 0.04f, 0.04f, 0.04f, //
    0.04f, 0.04f, 0.04f, 0.04f, 0.04f, //
};

__constant__ float sobelx_kernel[25] = {
    1, 2, 3, 2, 1,      //
    2, 3, 5, 3, 2,      //
    0, 0, 0, 0, 0,      //
    -2, -3, -5, -3, -2, //
    -1, -2, -3, -2, -1  //
};

__constant__ float sobely_kernel[25] = {
    -1, -2, 0, 2, 1, //
    -2, -3, 0, 3, 2, //
    -3, -5, 0, 5, 3, //
    -2, -3, 0, 3, 2, //
    -1, -2, 0, 2, 1  //
};

__global__ void SKR::kernels::getSmooth(unsigned char *r_in, unsigned char *g_in, unsigned char *b_in, int width, int height, unsigned char *r_out, unsigned char *g_out, unsigned char *b_out)
{
    unsigned int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (tindex < width * height)
    {
        runkernel5x5(r_in, g_in, b_in, width, height, smooth_kernel, tindex, r_out, g_out, b_out);
    }
}

__global__ void SKR::kernels::getSobel(unsigned char *in, int width, int height, float *sobelmag)
{
    unsigned int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (tindex < width * height)
    {
        float sobelx, sobely;
        runkernel5x5Sobel(in, &sobelx, width, height, sobelx_kernel, tindex);
        runkernel5x5Sobel(in, &sobely, width, height, sobely_kernel, tindex);
        sobelmag[tindex] = sqrtf(sobelx * sobelx + sobely * sobely);
    }
}

__global__ void SKR::kernels::getMax(float *data, float *maxv, unsigned int count)
{
    unsigned int tindex = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float sdata[MAX_CUDA_THREADS_PER_BLOCK];
    if (tindex < count)
    {
        sdata[threadIdx.x] = data[tindex];
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
        {
            if (threadIdx.x < stride)
            {
                float lhs = sdata[threadIdx.x];
                float rhs = sdata[threadIdx.x + stride];
                sdata[threadIdx.x] = MAX(lhs, rhs);
            }
            __syncthreads();
        }
    }
    if (tindex == 0)
    {
        *maxv = sdata[0];
    }
}

__global__ void SKR::kernels::getMin(float *data, float *minv, unsigned int count)
{
    unsigned int tindex = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float sdata[MAX_CUDA_THREADS_PER_BLOCK];
    if (tindex < count)
    {
        sdata[threadIdx.x] = data[tindex];
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
        {
            if (threadIdx.x < stride)
            {
                float lhs = sdata[threadIdx.x];
                float rhs = sdata[threadIdx.x + stride];
                sdata[threadIdx.x] = MIN(lhs, rhs);
            }
            __syncthreads();
        }
    }
    if (tindex == 0)
    {
        *minv = sdata[0];
    }
}

__global__ void SKR::kernels::getSobelEdges(float *sobelmag, float *minv, float *maxv, int width, int height, unsigned char threshold, unsigned char *out)
{
    unsigned int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (tindex < width * height)
    {
        float norm;
        norm = NORM(sobelmag[tindex], *minv, *maxv) * 255;
        out[tindex] = SET_UCHAR(norm);
        if (out[tindex] >= threshold)
        {
            out[tindex] = 255;
        }
        else
        {
            out[tindex] = 0;
        }
    }
}

SKR::imageprocess::imageprocess()
{
    CHECK_CUDA(cudaStreamCreate(&stream));
}

SKR::imageprocess::~imageprocess()
{
    CHECK_CUDA(cudaStreamDestroy(stream));
}

SKR::imageprocess &SKR::imageprocess::getInstance()
{
    static imageprocess ins;
    return ins;
}

void SKR::imageprocess::getNegative(jpegimage *img)
{
    MEASURE_TIME1;
    int blockn = (img->width * img->height + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    SKR::kernels::getNegative<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img->image.channel[0], img->image.channel[1], img->image.channel[2], img->width * img->height);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    MEASURE_TIME2("getNegative");
}

void SKR::imageprocess::getLighter(jpegimage *img, unsigned char value)
{
    MEASURE_TIME1;
    int blockn = (img->width * img->height + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    SKR::kernels::getLighter<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img->image.channel[0], img->image.channel[1], img->image.channel[2], value, img->width * img->height);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    MEASURE_TIME2("getLighter");
}

void SKR::imageprocess::getDarker(jpegimage *img, unsigned char value)
{
    MEASURE_TIME1;
    int blockn = (img->width * img->height + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    SKR::kernels::getDarker<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img->image.channel[0], img->image.channel[1], img->image.channel[2], value, img->width * img->height);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    MEASURE_TIME2("getDarker");
}

void SKR::imageprocess::getLowContrast(jpegimage *img, int value)
{
    MEASURE_TIME1;
    int blockn = (img->width * img->height + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    SKR::kernels::getLowContrast<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img->image.channel[0], img->image.channel[1], img->image.channel[2], value, img->width * img->height);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    MEASURE_TIME2("getLowContrast");
}

void SKR::imageprocess::getHighContrast(jpegimage *img, int value)
{
    MEASURE_TIME1;
    int blockn = (img->width * img->height + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    SKR::kernels::getHighContrast<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img->image.channel[0], img->image.channel[1], img->image.channel[2], value, img->width * img->height);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    MEASURE_TIME2("getHighContrast");
}

void SKR::imageprocess::getSmooth(jpegimage *img)
{
    MEASURE_TIME1;
    unsigned char *r_out = 0, *g_out = 0, *b_out = 0;
    CHECK_CUDA(cudaMalloc(&r_out, img->width * img->height));
    CHECK_CUDA(cudaMalloc(&g_out, img->width * img->height));
    CHECK_CUDA(cudaMalloc(&b_out, img->width * img->height));
    int blockn = (img->width * img->height + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    SKR::kernels::getSmooth<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img->image.channel[0], img->image.channel[1], img->image.channel[2], img->width, img->height, r_out, g_out, b_out);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaFree(img->image.channel[0]));
    CHECK_CUDA(cudaFree(img->image.channel[1]));
    CHECK_CUDA(cudaFree(img->image.channel[2]));
    img->image.channel[0] = r_out;
    img->image.channel[1] = g_out;
    img->image.channel[2] = b_out;
    MEASURE_TIME2("getSmooth");
}

void SKR::imageprocess::getGray(jpegimage *img)
{
    MEASURE_TIME1;
    int blockn = (img->width * img->height + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    SKR::kernels::getGray<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img->image.channel[0], img->image.channel[1], img->image.channel[2], img->width * img->height);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    MEASURE_TIME2("getGray");
}

void SKR::imageprocess::getSobelEdges(jpegimage *img, unsigned char threshold)
{
    MEASURE_TIME1;
    int blockn = (img->width * img->height + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    float *sobelmag, *minv, *maxv;
    CHECK_CUDA(cudaMalloc(&sobelmag, sizeof(float) * img->width * img->height));
    CHECK_CUDA(cudaMalloc(&minv, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&maxv, sizeof(float)));
    SKR::kernels::getSobel<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img->image.channel[0], img->width, img->height, sobelmag);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    SKR::kernels::getMin<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(sobelmag, minv, img->width * img->height);
    SKR::kernels::getMax<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(sobelmag, maxv, img->width * img->height);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    SKR::kernels::getSobelEdges<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(sobelmag, minv, maxv, img->width, img->height, threshold, img->image.channel[0]);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    cudaMemcpy(img->image.channel[1], img->image.channel[0], img->width * img->height, cudaMemcpyDeviceToDevice);
    cudaMemcpy(img->image.channel[2], img->image.channel[0], img->width * img->height, cudaMemcpyDeviceToDevice);
    CHECK_CUDA(cudaFree(sobelmag));
    CHECK_CUDA(cudaFree(minv));
    CHECK_CUDA(cudaFree(maxv));
    MEASURE_TIME2("getSobelEdges");
}