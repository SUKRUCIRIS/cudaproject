#include "templatematching.cuh"
#include "imageprocess.cuh"
#include "utility.cuh"
// ŞÜKRÜ ÇİRİŞ 2025

SKR::TemplateMatcherSSIMonEdge::TemplateMatcherSSIMonEdge(const std::string *targetfilenames, const int count)
{
    for (int i = 0; i < count; i++)
    {
        std::vector<unsigned char> *x = jpegde::getInstance().readJPEG(targetfilenames[i]);
        Image *y = jpegde::getInstance().decodeJPEG(*x);
        imageprocess::getInstance().getGray(y);
        imageprocess::getInstance().getSobelEdges(y, 70);
        targets.push_back(y);
        target_means.push_back(imageprocess::getInstance().getMean(y));
        target_variances.push_back(imageprocess::getInstance().getVariance(y, &target_means[i]));
        free(x);
    }
}
SKR::TemplateMatcherSSIMonEdge::~TemplateMatcherSSIMonEdge()
{
    for (int i = 0; i < targets.size(); i++)
    {
        jpegde::getInstance().freeJPEG(targets[i]);
    }
    targets.clear();
    target_means.clear();
    target_variances.clear();
}
SKR::vec2i SKR::TemplateMatcherSSIMonEdge::detectObject(const std::string &framefilename)
{
    vec2i biggest_similarity_point = {0, 0};
    float biggest_similarity = 0;
    std::vector<unsigned char> *x = jpegde::getInstance().readJPEG(framefilename);
    Image *frame = jpegde::getInstance().decodeJPEG(*x);
    free(x);
    imageprocess::getInstance().getGray(frame);
    imageprocess::getInstance().getSobelEdges(frame, 70);
    for (int ix = 0; ix < frame->width; ix++)
    {
        for (int iy = 0; iy < frame->height; iy++)
        {
            Image *candidate = imageprocess::getInstance()
                                   .extractCandidateForMatchingIndex(frame, targets[0]->width,
                                                                     targets[0]->height, GET_MINDEX(iy, ix, frame->width));
            if (candidate == 0)
            {
                continue;
            }
            float similarity = 0;
            for (int i = 0; i < targets.size(); i++)
            {
                similarity += imageprocess::getInstance()
                                  .getSSIMOneIsPreCalculated(candidate, targets[i], target_means[i],
                                                             target_variances[i]);
            }
            similarity /= targets.size();
            if (similarity > biggest_similarity)
            {
                biggest_similarity = similarity;
                biggest_similarity_point.x = ix;
                biggest_similarity_point.y = iy;
            }
            jpegde::getInstance().freeJPEG(candidate);
        }
    }
    jpegde::getInstance().freeJPEG(frame);
    return biggest_similarity_point;
}
SKR::vec2i SKR::TemplateMatcherSSIMonEdge::detectObjectBatch(const std::string &framefilename, const int batchsize,
                                                             bool earlyfinish)
{
    vec2i biggest_similarity_point = {0, 0};
    float biggest_similarity = 0;
    std::vector<unsigned char> *x = jpegde::getInstance().readJPEG(framefilename);
    Image *frame = jpegde::getInstance().decodeJPEG(*x);
    free(x);
    imageprocess::getInstance().getGray(frame);
    imageprocess::getInstance().getSobelEdges(frame, 70);

    unsigned char **out = 0;
    unsigned char **h_out = (unsigned char **)malloc(sizeof(unsigned char *) * batchsize);
    unsigned int pixelsize = targets[0]->width * targets[0]->height;
    for (int i = 0; i < batchsize; i++)
    {
        CHECK_CUDA(cudaMalloc(&(h_out[i]), sizeof(unsigned char) * pixelsize));
    }
    CHECK_CUDA(cudaMalloc(&out, sizeof(unsigned char *) * batchsize));
    CHECK_CUDA(cudaMemcpy(out, h_out, sizeof(unsigned char *) * batchsize, cudaMemcpyHostToDevice));
    unsigned char **img_out = (unsigned char **)malloc(sizeof(unsigned char *) * batchsize);

    int batchcount = ((frame->width * frame->height) + (batchsize - 1)) / batchsize;

    float *means = 0;
    float *variances = 0;
    CHECK_CUDA(cudaMalloc(&means, sizeof(float) * batchsize));
    CHECK_CUDA(cudaMalloc(&variances, sizeof(float) * batchsize));
    float **sum1 = 0, **sum2 = 0;
    sum1 = (float **)malloc(sizeof(float *) * batchsize);
    sum2 = (float **)malloc(sizeof(float *) * batchsize);
    unsigned int tmpblockn = ((pixelsize + (MAX_CUDA_THREADS_PER_BLOCK - 1)) / MAX_CUDA_THREADS_PER_BLOCK);
    for (int i = 0; i < batchsize; i++)
    {
        CHECK_CUDA(cudaMalloc(&(sum1[i]), sizeof(float) * tmpblockn));
        CHECK_CUDA(cudaMalloc(&(sum2[i]), sizeof(float) * tmpblockn));
    }
    float **covariances = 0, **ssim = 0;
    covariances = (float **)malloc(sizeof(float *) * targets.size());
    ssim = (float **)malloc(sizeof(float *) * targets.size());
    float **ssim_cpu = (float **)malloc(sizeof(float *) * targets.size());
    for (int i = 0; i < targets.size(); i++)
    {
        CHECK_CUDA(cudaMalloc(&(covariances[i]), sizeof(float) * batchsize));
        CHECK_CUDA(cudaMalloc(&(ssim[i]), sizeof(float) * batchsize));
        ssim_cpu[i] = (float *)malloc(sizeof(float) * batchsize);
    }
    float **sd1 = 0, *sd2 = 0, **mults = 0;
    sd1 = (float **)malloc(sizeof(float *) * batchsize);
    CHECK_CUDA(cudaMalloc(&sd2, sizeof(float) * pixelsize));
    mults = (float **)malloc(sizeof(float *) * batchsize);
    for (int i = 0; i < batchsize; i++)
    {
        CHECK_CUDA(cudaMalloc(&(sd1[i]), sizeof(float) * pixelsize));
        CHECK_CUDA(cudaMalloc(&(mults[i]), sizeof(float) * pixelsize));
    }

    for (int i = 0; i < batchcount; i++)
    {
        imageprocess::getInstance()
            .extractCandidatesForMatchingIndexMultiplePreAllocated(frame, targets[0]->width,
                                                                   targets[0]->height, i * batchsize,
                                                                   batchsize, h_out, out, img_out);

        imageprocess::getInstance().getMeanMultiplePreAllocated(img_out, pixelsize, batchsize, sum1, sum2, means);
        imageprocess::getInstance().getVarianceMultiplePreAllocated(img_out, pixelsize, batchsize, means,
                                                                    sd1, sum1, sum2, variances);
        for (int j = 0; j < targets.size(); j++)
        {
            imageprocess::getInstance().getCovarianceMultiplePreAllocated(img_out, targets[j]->image.channel[0], pixelsize,
                                                                          batchsize, means, target_means[j], sd1, sd2, mults,
                                                                          sum1, sum2, covariances[j]);
            imageprocess::getInstance().getSSIMMultiplePreAllocated(means, target_means[j], variances, target_variances[j],
                                                                    covariances[j], batchsize, ssim[j]);
            CHECK_CUDA(cudaMemcpy(ssim_cpu[j], ssim[j], sizeof(float) * batchsize, cudaMemcpyDeviceToHost));
        }

        for (int j = 0; j < batchsize; j++)
        {
            float similarity = 0;
            for (int k = 0; k < targets.size(); k++)
            {
                similarity += ssim_cpu[k][j];
            }
            similarity /= targets.size();
            if (similarity > biggest_similarity)
            {
                biggest_similarity = similarity;
                biggest_similarity_point.x = GET_MCOLUMN(i * batchsize + j, frame->width);
                biggest_similarity_point.y = GET_MROW(i * batchsize + j, frame->width);
            }
        }

        printf("Batch %d done\n", i);
        printf("Biggest similarity: %f\n", biggest_similarity);
        printf("Biggest similarity point: %d, %d\n", biggest_similarity_point.x, biggest_similarity_point.y);

        if (earlyfinish && biggest_similarity > (1.0f / targets.size()) + 0.05f)
        {
            break;
        }
    }

    for (int i = 0; i < batchsize; i++)
    {
        CHECK_CUDA(cudaFree(h_out[i]));
    }
    free(h_out);
    CHECK_CUDA(cudaFree(out));
    free(img_out);

    CHECK_CUDA(cudaFree(means));
    CHECK_CUDA(cudaFree(variances));
    for (int i = 0; i < batchsize; i++)
    {
        CHECK_CUDA(cudaFree(sum1[i]));
        CHECK_CUDA(cudaFree(sum2[i]));
        CHECK_CUDA(cudaFree(sd1[i]));
        CHECK_CUDA(cudaFree(mults[i]));
    }
    free(sum1);
    free(sum2);
    free(sd1);
    free(mults);
    for (int i = 0; i < targets.size(); i++)
    {
        CHECK_CUDA(cudaFree(covariances[i]));
        CHECK_CUDA(cudaFree(ssim[i]));
        free(ssim_cpu[i]);
    }
    free(covariances);
    free(ssim);
    free(ssim_cpu);
    CHECK_CUDA(cudaFree(sd2));

    return biggest_similarity_point;
}