#pragma once
#include "../../third_party/nvideo/Interface/nvcuvid.h"
#include "../../third_party/nvideo/Interface/cuviddec.h"
#include "../../third_party/nvideo/Interface/nvEncodeAPI.h"
#include <cuda.h>
// ŞÜKRÜ ÇİRİŞ 2024

namespace SKR
{
    class videode
    {
    private:
        videode();
        ~videode();
        CUcontext cuContext;

    public:
        videode(const videode &) = delete;
        videode &operator=(const videode &) = delete;

        // singletone
        static videode &getInstance();
    };
};