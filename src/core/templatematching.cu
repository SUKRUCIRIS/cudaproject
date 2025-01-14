#include "templatematching.cuh"
#include "imageprocess.cuh"
#include "utility.cuh"
// ŞÜKRÜ ÇİRİŞ 2025

SKR::TemplateMatcher::TemplateMatcher(const std::string *targetfilenames, const int count)
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
SKR::TemplateMatcher::~TemplateMatcher()
{
    for (int i = 0; i < targets.size(); i++)
    {
        jpegde::getInstance().freeJPEG(targets[i]);
    }
    targets.clear();
    target_means.clear();
    target_variances.clear();
}
SKR::vec2i SKR::TemplateMatcher::detectObject(const std::string &framefilename)
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
SKR::vec2i SKR::TemplateMatcher::detectObjectBatch(const std::string &framefilename, const int batchsize)
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
    for (int i = 0; i < batchsize; i++)
    {
        CHECK_CUDA(cudaMalloc(&(h_out[i]), sizeof(unsigned char) * targets[0]->width * targets[0]->height));
    }
    CHECK_CUDA(cudaMalloc(&out, sizeof(unsigned char *) * batchsize));
    CHECK_CUDA(cudaMemcpy(out, h_out, sizeof(unsigned char *) * batchsize, cudaMemcpyHostToDevice));
    Image *img_out = (Image *)malloc(sizeof(Image) * batchsize);

    int batchcount = (frame->width * frame->height) + (batchsize - 1) / batchsize;

    for (int i = 0; i < batchcount; i++)
    {
        printf("Batch %d started\n", i);
        imageprocess::getInstance()
            .extractCandidatesForMatchingIndexMultiplePreAllocated(frame, targets[0]->width,
                                                                   targets[0]->height, i * batchsize,
                                                                   batchsize, h_out, out, img_out);
        printf("Batch %d extracted\n", i);
        for (int j = 0; j < batchsize; j++)
        {
            if (img_out[j].image.channel[0] == 0)
            {
                continue;
            }
            float similarity = 0;
            for (int k = 0; k < targets.size(); k++)
            {
                similarity += imageprocess::getInstance()
                                  .getSSIMOneIsPreCalculated(&img_out[j], targets[k], target_means[k],
                                                             target_variances[k]);
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
    }

    for (int i = 0; i < batchsize; i++)
    {
        CHECK_CUDA(cudaFree(h_out[i]));
    }
    free(h_out);
    CHECK_CUDA(cudaFree(out));
    free(img_out);

    return biggest_similarity_point;
}