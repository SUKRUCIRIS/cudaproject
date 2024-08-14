#include "videode.cuh"
#include "utility.cuh"

SKR::videode::videode()
{
    CHECK_NVDEC(cuCtxCreate_v2(&cuContext, 0, 0));
}

SKR::videode::~videode()
{
    CHECK_NVDEC(cuCtxDestroy_v2(cuContext));
}

SKR::videode &SKR::videode::getInstance()
{
    static videode ins;
    return ins;
}