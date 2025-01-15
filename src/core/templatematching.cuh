#pragma once
// ŞÜKRÜ ÇİRİŞ 2025
#include "jpegde.cuh"

namespace SKR
{

    typedef struct vec2i
    {
        int x;
        int y;
    } vec2i;

    class TemplateMatcherSSIMonEdge
    {
    private:
        std::vector<Image *> targets;
        std::vector<float> target_means;
        std::vector<float> target_variances;

    public:
        // target is the object that we want to detect in the frame, count is the number of samples,
        // samples must be in the same size
        TemplateMatcherSSIMonEdge(const std::string *targetfilenames, const int count);
        ~TemplateMatcherSSIMonEdge();

        // too slow, don't use it, i will implement batching
        vec2i detectObject(const std::string &framefilename);

        // faster but still too slow. I figured out that SSIM is a slow algorithm for template matching
        // if your gpu doesn't have enough vram, you can decrease the batchsize
        vec2i detectObjectBatch(const std::string &framefilename, const int batchsize = 2000, bool earlyfinish = true);
    };

};