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

    class TemplateMatcher
    {
    private:
        std::vector<Image *> targets;
        std::vector<float> target_means;
        std::vector<float> target_variances;

    public:
        // target is the object that we want to detect in the frame, count is the number of samples,
        // samples must be in the same size
        TemplateMatcher(const std::string *targetfilenames, const int count);
        ~TemplateMatcher();

        // too slow, don't use it, i will implement batching
        vec2i detectObject(const std::string &framefilename);

        vec2i detectObjectBatch(const std::string &framefilename, const int batchsize = 10000);
    };

};