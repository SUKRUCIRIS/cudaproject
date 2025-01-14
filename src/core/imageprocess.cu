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
    if (tindex >= count)
    {
        sdata[threadIdx.x] = -FLT_MAX;
    }
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
    if (threadIdx.x == 0)
    {
        maxv[blockIdx.x] = sdata[0];
    }
}

__global__ void SKR::kernels::getMin(float *data, float *minv, unsigned int count)
{
    unsigned int tindex = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float sdata[MAX_CUDA_THREADS_PER_BLOCK];
    if (tindex >= count)
    {
        sdata[threadIdx.x] = FLT_MAX;
    }
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
    if (threadIdx.x == 0)
    {
        minv[blockIdx.x] = sdata[0];
    }
}

__global__ void SKR::kernels::getSum(unsigned char *data, float *sum, unsigned int count)
{
    unsigned int tindex = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float sdata[MAX_CUDA_THREADS_PER_BLOCK];
    if (tindex >= count)
    {
        sdata[threadIdx.x] = 0;
    }
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
                sdata[threadIdx.x] = lhs + rhs;
            }
            __syncthreads();
        }
    }
    if (threadIdx.x == 0)
    {
        sum[blockIdx.x] = sdata[0];
    }
}

__global__ void SKR::kernels::getSumFloat(float *data, float *sum, unsigned int count)
{
    unsigned int tindex = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float sdata[MAX_CUDA_THREADS_PER_BLOCK];
    if (tindex >= count)
    {
        sdata[threadIdx.x] = 0;
    }
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
                sdata[threadIdx.x] = lhs + rhs;
            }
            __syncthreads();
        }
    }
    if (threadIdx.x == 0)
    {
        sum[blockIdx.x] = sdata[0];
    }
}

__global__ void SKR::kernels::getMults(float *data1, float *data2, float *mults, unsigned int count)
{
    unsigned int tindex = blockDim.x * blockIdx.x + threadIdx.x;
    if (tindex < count)
    {
        mults[tindex] = data1[tindex] * data2[tindex];
    }
}

__global__ void SKR::kernels::getMultsOneSingle(float *data1, float data2, unsigned int count)
{
    unsigned int tindex = blockDim.x * blockIdx.x + threadIdx.x;
    if (tindex < count)
    {
        data1[tindex] = data1[tindex] * data2;
    }
}

__global__ void SKR::kernels::getNonSquaredDeviations(unsigned char *data, float mean, float *nsd, unsigned int count)
{
    unsigned int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (tindex < count)
    {
        nsd[tindex] = data[tindex] - mean;
    }
}

__global__ void SKR::kernels::getSquaredDeviations(unsigned char *data, float mean, float *sd, unsigned int count)
{
    unsigned int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (tindex < count)
    {
        float diff = data[tindex] - mean;
        sd[tindex] = diff * diff;
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

__global__ void SKR::kernels::splitSingleChannel(unsigned char *in, unsigned char **out, int width, int height, int splitwidth, int splitheight)
{
    unsigned int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (tindex < width * height)
    {
        unsigned int column = GET_MCOLUMN(tindex, width);
        unsigned int row = GET_MROW(tindex, width);
        unsigned int splitcolumn = column / splitwidth;
        unsigned int splitrow = row / splitheight;
        unsigned int splitindex = GET_MINDEX(splitrow, splitcolumn, width / splitwidth);
        unsigned int local_column = column % splitwidth;
        unsigned int local_row = row % splitheight;
        unsigned int local_index = GET_MINDEX(local_row, local_column, splitwidth);
        out[splitindex][local_index] = in[tindex];
    }
}

__global__ void SKR::kernels::splitSingleChannelTemplate(unsigned char *in, unsigned char **out, int width, int height, int templateWidth, int templateHeight)
{
    unsigned int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (tindex < width * height)
    {
        unsigned int column = GET_MCOLUMN(tindex, width);
        unsigned int row = GET_MROW(tindex, width);
        int templateIndex = GET_MINDEX(row, column, width - templateWidth + 1);

        if (column + templateWidth <= width && row + templateHeight <= height)
        {
            for (int i = 0; i < templateHeight; i++)
            {
                for (int j = 0; j < templateWidth; j++)
                {
                    unsigned int localIndex = GET_MINDEX(i, j, templateWidth);
                    out[templateIndex][localIndex] = in[GET_MINDEX(row + i, column + j, width)];
                }
            }
        }
    }
}

__global__ void SKR::kernels::splitSingleChannelTemplateIndex(unsigned char *in, unsigned char *out, int width, int height, int splitwidth, int splitheight, int index)
{
    unsigned int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (tindex < splitwidth * splitheight)
    {
        unsigned int localcolumn = GET_MCOLUMN(tindex, splitwidth);
        unsigned int localrow = GET_MROW(tindex, splitwidth);
        unsigned int column = localcolumn + GET_MCOLUMN(index, width);
        unsigned int row = localrow + GET_MROW(index, width);
        out[tindex] = in[GET_MINDEX(row, column, width)];
    }
}

__global__ void SKR::kernels::splitSingleChannelTemplateIndexMultiple(unsigned char *in, unsigned char **out, int width, int height, int splitwidth, int splitheight, int index, int count)
{
    unsigned int tindex = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int batchIndex = tindex / (splitwidth * splitheight);
    unsigned int localIndex = tindex % (splitwidth * splitheight);

    if (batchIndex < count)
    {
        unsigned int globalIndex = index + batchIndex;
        unsigned int globalColumn = GET_MCOLUMN(globalIndex, width);
        unsigned int globalRow = GET_MROW(globalIndex, width);

        if (globalColumn + splitwidth <= width && globalRow + splitheight <= height)
        {
            unsigned int localcolumn = GET_MCOLUMN(localIndex, splitwidth);
            unsigned int localrow = GET_MROW(localIndex, splitwidth);
            unsigned int column = localcolumn + globalColumn;
            unsigned int row = localrow + globalRow;
            out[batchIndex][localIndex] = in[GET_MINDEX(row, column, width)];
        }
        else
        {
            out[batchIndex][localIndex] = 0;
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

void SKR::imageprocess::getNegative(Image *img)
{
    MEASURE_TIME1;
    int blockn = (img->width * img->height + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    SKR::kernels::getNegative<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img->image.channel[0], img->image.channel[1], img->image.channel[2], img->width * img->height);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    MEASURE_TIME2("getNegative");
}

void SKR::imageprocess::getLighter(Image *img, unsigned char value)
{
    MEASURE_TIME1;
    int blockn = (img->width * img->height + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    SKR::kernels::getLighter<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img->image.channel[0], img->image.channel[1], img->image.channel[2], value, img->width * img->height);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    MEASURE_TIME2("getLighter");
}

void SKR::imageprocess::getDarker(Image *img, unsigned char value)
{
    MEASURE_TIME1;
    int blockn = (img->width * img->height + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    SKR::kernels::getDarker<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img->image.channel[0], img->image.channel[1], img->image.channel[2], value, img->width * img->height);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    MEASURE_TIME2("getDarker");
}

void SKR::imageprocess::getLowContrast(Image *img, int value)
{
    MEASURE_TIME1;
    int blockn = (img->width * img->height + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    SKR::kernels::getLowContrast<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img->image.channel[0], img->image.channel[1], img->image.channel[2], value, img->width * img->height);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    MEASURE_TIME2("getLowContrast");
}

void SKR::imageprocess::getHighContrast(Image *img, int value)
{
    MEASURE_TIME1;
    int blockn = (img->width * img->height + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    SKR::kernels::getHighContrast<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img->image.channel[0], img->image.channel[1], img->image.channel[2], value, img->width * img->height);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    MEASURE_TIME2("getHighContrast");
}

void SKR::imageprocess::getSmooth(Image *img)
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

void SKR::imageprocess::getGray(Image *img)
{
    MEASURE_TIME1;
    int blockn = (img->width * img->height + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    SKR::kernels::getGray<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img->image.channel[0], img->image.channel[1], img->image.channel[2], img->width * img->height);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    MEASURE_TIME2("getGray");
}

void SKR::imageprocess::getSobelEdges(Image *img, unsigned char threshold)
{
    MEASURE_TIME1;
    int blockn = (img->width * img->height + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    float *sobelmag, *minv, *maxv;
    CHECK_CUDA(cudaMalloc(&sobelmag, sizeof(float) * img->width * img->height));
    CHECK_CUDA(cudaMalloc(&minv, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&maxv, sizeof(float)));
    SKR::kernels::getSobel<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img->image.channel[0], img->width, img->height, sobelmag);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    float tmpminv = getMin(sobelmag, img->width * img->height);
    float tmpmaxv = getMax(sobelmag, img->width * img->height);
    CHECK_CUDA(cudaMemcpy(minv, &tmpminv, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(maxv, &tmpmaxv, sizeof(float), cudaMemcpyHostToDevice));
    SKR::kernels::getSobelEdges<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(sobelmag, minv, maxv, img->width, img->height, threshold, img->image.channel[0]);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMemcpy(img->image.channel[1], img->image.channel[0], img->width * img->height, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(img->image.channel[2], img->image.channel[0], img->width * img->height, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaFree(sobelmag));
    CHECK_CUDA(cudaFree(minv));
    CHECK_CUDA(cudaFree(maxv));
    MEASURE_TIME2("getSobelEdges");
}

float SKR::imageprocess::getSum(Image *img)
{
    return getSum(img->image.channel[0], img->width * img->height);
}

float SKR::imageprocess::getSum(unsigned char *img, unsigned int count)
{
    MEASURE_TIME1;
    float res = 0;
    int blockn = (count + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    float *result = 0;
    CHECK_CUDA(cudaMalloc(&result, sizeof(float) * blockn));
    SKR::kernels::getSum<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img, result, count);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    // add recursively block results
    float *result2 = 0;
    if (blockn > 1)
    {
        CHECK_CUDA(cudaMalloc(&result2, sizeof(float) * blockn));
    }
    else
    {
        CHECK_CUDA(cudaMemcpy(&res, result, sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaFree(result));
        MEASURE_TIME2("getSum");
        return res;
    }
    int before_blockn = blockn;
    blockn = (blockn + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    while (1)
    {
        SKR::kernels::getSumFloat<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(result, result2, before_blockn);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        CHECK_CUDA(cudaMemcpy(result, result2, sizeof(float) * blockn, cudaMemcpyDeviceToDevice));
        if (blockn == 1)
        {
            CHECK_CUDA(cudaMemcpy(&res, result2, sizeof(float), cudaMemcpyDeviceToHost));
            break;
        }
        before_blockn = blockn;
        if (blockn <= MAX_CUDA_THREADS_PER_BLOCK)
        {
            blockn = 1;
        }
        else
        {
            blockn = (blockn + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
        }
    }
    CHECK_CUDA(cudaFree(result));
    CHECK_CUDA(cudaFree(result2));
    MEASURE_TIME2("getSum");
    return res;
}

void SKR::imageprocess::getSumMultiplePreAllocated(unsigned char **img, unsigned int pixel_count,
                                                   unsigned int batch_count, float **sum1,
                                                   float **sum2, float *result)
{
    MEASURE_TIME1;
    int blockn = (pixel_count + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    for (int i = 0; i < batch_count; i++)
    {
        SKR::kernels::getSum<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img[i], sum1[i], pixel_count);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    if (blockn <= 1)
    {
        for (int i = 0; i < batch_count; i++)
        {
            CHECK_CUDA(cudaMemcpy(&(result[i]), sum1[i], sizeof(float), cudaMemcpyDeviceToHost));
        }
        MEASURE_TIME2("getSumMultiplePreAllocated");
    }

    int before_blockn = blockn;
    blockn = (blockn + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;

    while (1)
    {
        for (int i = 0; i < batch_count; i++)
        {
            SKR::kernels::getSumFloat<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(sum1[i], sum2[i], before_blockn);
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));
        for (int i = 0; i < batch_count; i++)
        {
            CHECK_CUDA(cudaMemcpy(sum1[i], sum2[i], sizeof(float) * blockn, cudaMemcpyDeviceToDevice));
        }
        if (blockn == 1)
        {
            for (int i = 0; i < batch_count; i++)
            {
                CHECK_CUDA(cudaMemcpy(&(result[i]), sum1[i], sizeof(float), cudaMemcpyDeviceToHost));
            }
            break;
        }
        before_blockn = blockn;
        if (blockn <= MAX_CUDA_THREADS_PER_BLOCK)
        {
            blockn = 1;
        }
        else
        {
            blockn = (blockn + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
        }
    }

    MEASURE_TIME2("getSumMultiplePreAllocated");
}

float SKR::imageprocess::getSum(float *img, unsigned int count)
{
    MEASURE_TIME1;
    float res = 0;
    int blockn = (count + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    float *result = 0;
    CHECK_CUDA(cudaMalloc(&result, sizeof(float) * blockn));
    SKR::kernels::getSumFloat<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img, result, count);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    // add recursively block results
    float *result2 = 0;
    if (blockn > 1)
    {
        CHECK_CUDA(cudaMalloc(&result2, sizeof(float) * blockn));
    }
    else
    {
        CHECK_CUDA(cudaMemcpy(&res, result, sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaFree(result));
        MEASURE_TIME2("getSum");
        return res;
    }
    int before_blockn = blockn;
    blockn = (blockn + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    while (1)
    {
        SKR::kernels::getSumFloat<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(result, result2, before_blockn);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        CHECK_CUDA(cudaMemcpy(result, result2, sizeof(float) * blockn, cudaMemcpyDeviceToDevice));
        if (blockn == 1)
        {
            CHECK_CUDA(cudaMemcpy(&res, result2, sizeof(float), cudaMemcpyDeviceToHost));
            break;
        }
        before_blockn = blockn;
        if (blockn <= MAX_CUDA_THREADS_PER_BLOCK)
        {
            blockn = 1;
        }
        else
        {
            blockn = (blockn + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
        }
    }
    CHECK_CUDA(cudaFree(result));
    CHECK_CUDA(cudaFree(result2));
    MEASURE_TIME2("getSum");
    return res;
}

void SKR::imageprocess::getSumMultiplePreAllocated(float **img, unsigned int pixel_count, unsigned int batch_count,
                                                   float **sum1, float **sum2, float *result)
{
    MEASURE_TIME1;
    int blockn = (pixel_count + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    for (int i = 0; i < batch_count; i++)
    {
        SKR::kernels::getSumFloat<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img[i], sum1[i], pixel_count);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    if (blockn <= 1)
    {
        for (int i = 0; i < batch_count; i++)
        {
            CHECK_CUDA(cudaMemcpy(&(result[i]), sum1[i], sizeof(float), cudaMemcpyDeviceToHost));
        }
        MEASURE_TIME2("getSumMultiplePreAllocated");
    }

    int before_blockn = blockn;
    blockn = (blockn + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;

    while (1)
    {
        for (int i = 0; i < batch_count; i++)
        {
            SKR::kernels::getSumFloat<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(sum1[i], sum2[i], before_blockn);
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));
        for (int i = 0; i < batch_count; i++)
        {
            CHECK_CUDA(cudaMemcpy(sum1[i], sum2[i], sizeof(float) * blockn, cudaMemcpyDeviceToDevice));
        }
        if (blockn == 1)
        {
            for (int i = 0; i < batch_count; i++)
            {
                CHECK_CUDA(cudaMemcpy(&(result[i]), sum2[i], sizeof(float), cudaMemcpyDeviceToHost));
            }
            break;
        }
        before_blockn = blockn;
        if (blockn <= MAX_CUDA_THREADS_PER_BLOCK)
        {
            blockn = 1;
        }
        else
        {
            blockn = (blockn + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
        }
    }

    MEASURE_TIME2("getSumMultiplePreAllocated");
}

float SKR::imageprocess::getMin(float *img, unsigned int count)
{
    MEASURE_TIME1;
    float res = 0;
    int blockn = (count + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    float *result = 0;
    CHECK_CUDA(cudaMalloc(&result, sizeof(float) * blockn));
    SKR::kernels::getMin<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img, result, count);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    // add recursively block results
    float *result2 = 0;
    if (blockn > 1)
    {
        CHECK_CUDA(cudaMalloc(&result2, sizeof(float) * blockn));
    }
    else
    {
        CHECK_CUDA(cudaMemcpy(&res, result, sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaFree(result));
        MEASURE_TIME2("getMin");
        return res;
    }
    int before_blockn = blockn;
    blockn = (blockn + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    while (1)
    {
        SKR::kernels::getMin<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(result, result2, before_blockn);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        CHECK_CUDA(cudaMemcpy(result, result2, sizeof(float) * blockn, cudaMemcpyDeviceToDevice));
        if (blockn == 1)
        {
            CHECK_CUDA(cudaMemcpy(&res, result2, sizeof(float), cudaMemcpyDeviceToHost));
            break;
        }
        before_blockn = blockn;
        if (blockn <= MAX_CUDA_THREADS_PER_BLOCK)
        {
            blockn = 1;
        }
        else
        {
            blockn = (blockn + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
        }
    }
    CHECK_CUDA(cudaFree(result));
    CHECK_CUDA(cudaFree(result2));
    MEASURE_TIME2("getMin");
    return res;
}

float SKR::imageprocess::getMax(float *img, unsigned int count)
{
    MEASURE_TIME1;
    float res = 0;
    int blockn = (count + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    float *result = 0;
    CHECK_CUDA(cudaMalloc(&result, sizeof(float) * blockn));
    SKR::kernels::getMax<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img, result, count);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    // add recursively block results
    float *result2 = 0;
    if (blockn > 1)
    {
        CHECK_CUDA(cudaMalloc(&result2, sizeof(float) * blockn));
    }
    else
    {
        CHECK_CUDA(cudaMemcpy(&res, result, sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaFree(result));
        MEASURE_TIME2("getMax");
        return res;
    }
    int before_blockn = blockn;
    blockn = (blockn + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    while (1)
    {
        SKR::kernels::getMax<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(result, result2, before_blockn);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        CHECK_CUDA(cudaMemcpy(result, result2, sizeof(float) * blockn, cudaMemcpyDeviceToDevice));
        if (blockn == 1)
        {
            CHECK_CUDA(cudaMemcpy(&res, result2, sizeof(float), cudaMemcpyDeviceToHost));
            break;
        }
        before_blockn = blockn;
        if (blockn <= MAX_CUDA_THREADS_PER_BLOCK)
        {
            blockn = 1;
        }
        else
        {
            blockn = (blockn + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
        }
    }
    CHECK_CUDA(cudaFree(result));
    CHECK_CUDA(cudaFree(result2));
    MEASURE_TIME2("getMax");
    return res;
}

float SKR::imageprocess::getMean(Image *img)
{
    return getSum(img) / (img->width * img->height);
}

void SKR::imageprocess::getMeanMultiplePreAllocated(unsigned char **img, unsigned int pixel_count,
                                                    unsigned int batch_count, float **sum1,
                                                    float **sum2, float *sum3, float *result)
{
    getSumMultiplePreAllocated(img, pixel_count, batch_count, sum1, sum2, result);
    CHECK_CUDA(cudaMemcpy(sum3, result, sizeof(float) * batch_count, cudaMemcpyHostToDevice));
    int blockn = (batch_count + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    SKR::kernels::getMultsOneSingle<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(sum3, 1.0f / (float)pixel_count, batch_count);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMemcpy(result, sum3, sizeof(float) * batch_count, cudaMemcpyDeviceToHost));
}

std::vector<SKR::Image *> *SKR::imageprocess::splitSingleChannel(Image *img, int splitwidth, int splitheight)
{
    if (img->image.channel[0] == 0 || img->width % splitwidth != 0 || img->height % splitheight != 0 ||
        splitwidth == 0 || splitheight == 0)
    {
        std::cout << "Invalid split parameters" << std::endl;
        return 0;
    }
    MEASURE_TIME1;
    std::vector<Image *> *res = new std::vector<Image *>();

    int splitcount = (img->width / splitwidth) * (img->height / splitheight);
    int splitsize = splitwidth * splitheight;
    unsigned char **out = 0;
    unsigned char **h_out = (unsigned char **)malloc(sizeof(unsigned char *) * splitcount);
    for (int i = 0; i < splitcount; i++)
    {
        CHECK_CUDA(cudaMalloc(&(h_out[i]), sizeof(unsigned char) * splitsize));
    }
    CHECK_CUDA(cudaMalloc(&out, sizeof(unsigned char *) * splitcount));
    CHECK_CUDA(cudaMemcpy(out, h_out, sizeof(unsigned char *) * splitcount, cudaMemcpyHostToDevice));
    int blockn = (img->width * img->height + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    SKR::kernels::splitSingleChannel<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img->image.channel[0], out, img->width, img->height, splitwidth, splitheight);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (int i = 0; i < splitcount; i++)
    {
        Image *x = jpegde::getInstance().createImage(h_out[i], splitwidth, splitheight);
        res->push_back(x);
    }
    CHECK_CUDA(cudaFree(out));
    free(h_out);

    MEASURE_TIME2("splitSingleChannel");
    return res;
}

float SKR::imageprocess::getVariance(Image *img, float *pre_mean)
{
    MEASURE_TIME1;
    float mean = 0;
    if (pre_mean == 0)
    {
        mean = getMean(img);
    }
    else
    {
        mean = *pre_mean;
    }
    float *sd = 0;
    CHECK_CUDA(cudaMalloc(&sd, sizeof(float) * img->width * img->height));
    int blockn = (img->width * img->height + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    SKR::kernels::getSquaredDeviations<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img->image.channel[0], mean, sd, img->width * img->height);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    float res = getSum(sd, img->width * img->height) / ((img->width * img->height) - 1);
    CHECK_CUDA(cudaFree(sd));
    MEASURE_TIME2("getVariance");
    return res;
}

float SKR::imageprocess::getStandardDeviation(Image *img, float *pre_mean)
{
    return sqrtf(getVariance(img, pre_mean));
}

float SKR::imageprocess::getCovariance(Image *img1, Image *img2, float *pre_mean1, float *pre_mean2)
{
    if (img1->width != img2->width || img1->height != img2->height)
    {
        std::cout << "Images must have same dimensions for covariance" << std::endl;
        return 0;
    }
    MEASURE_TIME1;
    float mean1 = 0, mean2 = 0;
    if (pre_mean1 == 0)
    {
        mean1 = getMean(img1);
    }
    else
    {
        mean1 = *pre_mean1;
    }
    if (pre_mean2 == 0)
    {
        mean2 = getMean(img2);
    }
    else
    {
        mean2 = *pre_mean2;
    }
    float *sd1 = 0, *sd2 = 0;
    CHECK_CUDA(cudaMalloc(&sd1, sizeof(float) * img1->width * img1->height));
    CHECK_CUDA(cudaMalloc(&sd2, sizeof(float) * img2->width * img2->height));
    int blockn = (img1->width * img1->height + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    SKR::kernels::getNonSquaredDeviations<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img1->image.channel[0], mean1, sd1, img1->width * img1->height);
    SKR::kernels::getNonSquaredDeviations<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img2->image.channel[0], mean2, sd2, img2->width * img2->height);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    float *mults = 0;
    CHECK_CUDA(cudaMalloc(&mults, sizeof(float) * img1->width * img1->height));
    SKR::kernels::getMults<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(sd1, sd2, mults, img1->width * img1->height);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    float res = getSum(mults, img1->width * img1->height) / (img1->width * img1->height - 1);
    CHECK_CUDA(cudaFree(sd1));
    CHECK_CUDA(cudaFree(sd2));
    CHECK_CUDA(cudaFree(mults));
    MEASURE_TIME2("getCovariance");
    return res;
}

float SKR::imageprocess::getSSIM(Image *img1, Image *img2, float K1, float K2, float L, float *pre_mean1, float *pre_mean2)
{
    if (img1->width != img2->width || img1->height != img2->height)
    {
        std::cout << "Images must have same dimensions for SSIM" << std::endl;
        return 0;
    }
    MEASURE_TIME1;
    float C1 = (K1 * L) * (K1 * L);
    float C2 = (K2 * L) * (K2 * L);
    float mean1 = 0, mean2 = 0;
    if (pre_mean1 == 0)
    {
        mean1 = getMean(img1);
    }
    else
    {
        mean1 = *pre_mean1;
    }
    if (pre_mean2 == 0)
    {
        mean2 = getMean(img2);
    }
    else
    {
        mean2 = *pre_mean2;
    }
    float variance1 = getVariance(img1, &mean1);
    float variance2 = getVariance(img2, &mean2);
    float covariance = getCovariance(img1, img2, &mean1, &mean2);
    float ssim = ((2 * mean1 * mean2 + C1) * (2 * covariance + C2)) /
                 ((mean1 * mean1 + mean2 * mean2 + C1) * (variance1 + variance2 + C2));
    MEASURE_TIME2("getSSIM");
    return ssim;
}

float SKR::imageprocess::getSSIMOneIsPreCalculated(Image *img1, Image *img2, float pre_mean2, float pre_variance2,
                                                   float K1, float K2, float L, float *pre_mean1)
{
    if (img1->width != img2->width || img1->height != img2->height)
    {
        std::cout << "Images must have same dimensions for SSIM" << std::endl;
        return 0;
    }
    MEASURE_TIME1;
    float C1 = (K1 * L) * (K1 * L);
    float C2 = (K2 * L) * (K2 * L);
    float mean1 = 0;
    if (pre_mean1 == 0)
    {
        mean1 = getMean(img1);
    }
    else
    {
        mean1 = *pre_mean1;
    }
    float variance1 = getVariance(img1, &mean1);
    float covariance = getCovariance(img1, img2, &mean1, &pre_mean2);
    float ssim = ((2 * mean1 * pre_mean2 + C1) * (2 * covariance + C2)) /
                 ((mean1 * mean1 + pre_mean2 * pre_mean2 + C1) * (variance1 + pre_variance2 + C2));
    MEASURE_TIME2("getSSIM");
    return ssim;
}

std::vector<SKR::Image *> *SKR::imageprocess::extractCandidatesForMatching(Image *img, int splitwidth, int splitheight)
{
    if (img->image.channel[0] == 0 || img->width < splitwidth || img->height < splitheight || splitwidth == 0 || splitheight == 0)
    {
        std::cout << "Invalid split parameters" << std::endl;
        return 0;
    }
    MEASURE_TIME1;
    std::vector<Image *> *res = new std::vector<Image *>();

    int splitcount = (img->width - splitwidth + 1) * (img->height - splitheight + 1);
    int splitsize = splitwidth * splitheight;
    unsigned char **out = 0;
    unsigned char **h_out = (unsigned char **)malloc(sizeof(unsigned char *) * splitcount);
    for (int i = 0; i < splitcount; i++)
    {
        CHECK_CUDA(cudaMalloc(&(h_out[i]), sizeof(unsigned char) * splitsize));
    }
    CHECK_CUDA(cudaMalloc(&out, sizeof(unsigned char *) * splitcount));
    CHECK_CUDA(cudaMemcpy(out, h_out, sizeof(unsigned char *) * splitcount, cudaMemcpyHostToDevice));
    int blockn = (img->width * img->height + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    SKR::kernels::splitSingleChannelTemplate<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img->image.channel[0], out, img->width, img->height, splitwidth, splitheight);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (int i = 0; i < splitcount; i++)
    {
        Image *x = jpegde::getInstance().createImage(h_out[i], splitwidth, splitheight);
        res->push_back(x);
    }
    CHECK_CUDA(cudaFree(out));
    free(h_out);

    MEASURE_TIME2("extractCandidatesForMatching");
    return res;
}

SKR::Image *SKR::imageprocess::extractCandidateForMatchingIndex(Image *img, int splitwidth, int splitheight, int index)
{
    if (img->image.channel[0] == 0 || img->width < splitwidth ||
        img->height < splitheight || splitwidth == 0 || splitheight == 0 ||
        img->width - splitwidth < GET_MCOLUMN(index, img->width) ||
        img->height - splitheight < GET_MROW(index, img->width))
    {
        std::cout << "Invalid split parameters" << std::endl;
        return 0;
    }
    MEASURE_TIME1;
    unsigned char *out = 0;
    CHECK_CUDA(cudaMalloc(&out, sizeof(unsigned char) * splitwidth * splitheight));
    int blockn = (splitwidth * splitheight + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    SKR::kernels::splitSingleChannelTemplateIndex<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img->image.channel[0], out, img->width, img->height, splitwidth, splitheight, index);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    Image *res = jpegde::getInstance().createImage(out, splitwidth, splitheight);
    MEASURE_TIME2("extractCandidateForMatchingIndex");
    return res;
}

std::vector<SKR::Image *> *SKR::imageprocess::extractCandidatesForMatchingIndexMultiple(Image *img, int splitwidth, int splitheight,
                                                                                        int index, int count)
{
    if (img->image.channel[0] == 0 || img->width < splitwidth || img->height < splitheight || splitwidth == 0 || splitheight == 0)
    {
        std::cout << "Invalid split parameters" << std::endl;
        return 0;
    }
    MEASURE_TIME1;
    std::vector<Image *> *res = new std::vector<Image *>();

    int splitsize = splitwidth * splitheight;
    unsigned char **out = 0;
    unsigned char **h_out = (unsigned char **)malloc(sizeof(unsigned char *) * count);
    for (int i = 0; i < count; i++)
    {
        CHECK_CUDA(cudaMalloc(&(h_out[i]), sizeof(unsigned char) * splitsize));
    }
    CHECK_CUDA(cudaMalloc(&out, sizeof(unsigned char *) * count));
    CHECK_CUDA(cudaMemcpy(out, h_out, sizeof(unsigned char *) * count, cudaMemcpyHostToDevice));
    int blockn = (splitsize * count + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    SKR::kernels::splitSingleChannelTemplateIndexMultiple<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img->image.channel[0], out, img->width, img->height, splitwidth, splitheight, index, count);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (int i = 0; i < count; i++)
    {
        Image *x = jpegde::getInstance().createImage(h_out[i], splitwidth, splitheight);
        res->push_back(x);
    }
    CHECK_CUDA(cudaFree(out));
    free(h_out);

    MEASURE_TIME2("extractCandidatesForMatchingIndexMultiple");
    return res;
}

void SKR::imageprocess::extractCandidatesForMatchingIndexMultiplePreAllocated(Image *img, int splitwidth, int splitheight,
                                                                              int index, int count, unsigned char **h_out,
                                                                              unsigned char **out, Image *img_out)
{
    if (img->image.channel[0] == 0 || img->width < splitwidth || img->height < splitheight || splitwidth == 0 || splitheight == 0)
    {
        std::cout << "Invalid split parameters" << std::endl;
        return;
    }
    MEASURE_TIME1;
    int blockn = (splitwidth * splitheight * count + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK;
    SKR::kernels::splitSingleChannelTemplateIndexMultiple<<<blockn, MAX_CUDA_THREADS_PER_BLOCK, 0, stream>>>(img->image.channel[0], out, img->width, img->height, splitwidth, splitheight, index, count);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (int i = 0; i < count; i++)
    {
        img_out[i].image.channel[0] = h_out[i];
        img_out[i].image.channel[1] = h_out[i];
        img_out[i].image.channel[2] = h_out[i];
        img_out[i].image.channel[3] = 0;
        img_out[i].width = splitwidth;
        img_out[i].height = splitheight;
        img_out[i].image.pitch[0] = splitwidth;
        img_out[i].image.pitch[1] = splitwidth;
        img_out[i].image.pitch[2] = splitwidth;
    }
    MEASURE_TIME2("extractCandidatesForMatchingIndexMultiplePreAllocated");
}