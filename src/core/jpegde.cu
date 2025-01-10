#include "jpegde.cuh"
#include "utility.cuh"
// ŞÜKRÜ ÇİRİŞ 2024

SKR::jpegde::jpegde(void)
{
    CHECK_NVJPEG(nvjpegCreateSimple(&handle))
    CHECK_NVJPEG(nvjpegJpegStateCreate(handle, &state))
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_NVJPEG(nvjpegEncoderStateCreate(handle, &enc_state, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsCreate(handle, &enc_params, stream));
}

SKR::jpegde &SKR::jpegde::getInstance()
{
    static jpegde ins;
    return ins;
}

std::vector<unsigned char> *SKR::jpegde::readJPEG(const std::string &filename)
{
    MEASURE_TIME1;
    std::vector<unsigned char> *buffer = new std::vector<unsigned char>;
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file)
    {
        std::cout << "Failed to open file for reading: " << filename << std::endl;
        exit(-1);
    }

    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    buffer->resize(size);
    if (!file.read(reinterpret_cast<char *>(buffer->data()), size))
    {
        std::cout << "Failed to read the file: " << filename << std::endl;
        exit(-1);
    }
    MEASURE_TIME2("readJPEG");
    return buffer;
}

SKR::Image *SKR::jpegde::decodeJPEG(const std::vector<unsigned char> &jpeg_buffer)
{
    MEASURE_TIME1;
    int nComponents;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];

    Image *output_image = new Image;

    nvjpegChromaSubsampling_t subsampling;

    CHECK_NVJPEG(nvjpegGetImageInfo(handle, jpeg_buffer.data(), jpeg_buffer.size(), &nComponents, &subsampling, widths, heights));

    output_image->width = widths[0];
    output_image->height = heights[0];
    int size = heights[0] * widths[0];
    for (int i = 0; i < nComponents; i++)
    {
        CHECK_CUDA(cudaMalloc(&output_image->image.channel[i], size));
        output_image->image.pitch[i] = widths[0];
    }
    for (int i = nComponents; i < NVJPEG_MAX_COMPONENT; i++)
    {
        output_image->image.channel[i] = 0;
        output_image->image.pitch[i] = 0;
    }

    CHECK_NVJPEG(nvjpegDecode(handle, state, jpeg_buffer.data(), jpeg_buffer.size(), NVJPEG_OUTPUT_RGB, &output_image->image, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));
    MEASURE_TIME2("decodeJPEG");
    return output_image;
}

SKR::jpegde::~jpegde()
{
    CHECK_NVJPEG(nvjpegEncoderParamsDestroy(enc_params));
    CHECK_NVJPEG(nvjpegEncoderStateDestroy(enc_state));
    CHECK_NVJPEG(nvjpegJpegStateDestroy(state));
    CHECK_NVJPEG(nvjpegDestroy(handle));
    CHECK_CUDA(cudaStreamDestroy(stream));
}

void SKR::jpegde::freeJPEG(Image *image)
{
    MEASURE_TIME1;
    if (image->image.channel[0] != image->image.channel[1])
    {
        for (int i = 0; i < NVJPEG_MAX_COMPONENT; i++)
        {
            CHECK_CUDA(cudaFree(image->image.channel[i]));
        }
    }
    else
    {
        // single channel
        CHECK_CUDA(cudaFree(image->image.channel[0]));
    }

    delete image;
    MEASURE_TIME2("freeJPEG");
}

std::vector<unsigned char> *SKR::jpegde::encodeJPEG(const Image *image, const int quality, const bool isgray)
{
    MEASURE_TIME1;
    if (isgray)
    {
        CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(enc_params, NVJPEG_CSS_GRAY, stream));
    }
    else
    {
        CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(enc_params, NVJPEG_CSS_420, stream));
    }
    CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(enc_params, quality, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_NVJPEG(nvjpegEncodeImage(handle, enc_state, enc_params, &(image->image),
                                   NVJPEG_INPUT_RGB, image->width, image->height, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    size_t length = 0;
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle, enc_state, 0, &length, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<unsigned char> *encoded = new std::vector<unsigned char>(length);
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle, enc_state, encoded->data(), &length, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));
    MEASURE_TIME2("encodeJPEG");
    return encoded;
}

void SKR::jpegde::writeJPEG(const std::string &filename, const std::vector<unsigned char> &jpeg_buffer)
{
    MEASURE_TIME1;
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file)
    {
        std::cout << "Failed to open file for writing: " << filename << std::endl;
        exit(-1);
    }
    file.write(reinterpret_cast<const char *>(jpeg_buffer.data()), jpeg_buffer.size());
    if (!file)
    {
        std::cout << "Failed to write data to file: " << filename << std::endl;
        exit(-1);
    }
    file.close();
    MEASURE_TIME2("writeJPEG");
}

SKR::Image *SKR::jpegde::createImage(unsigned char *r, unsigned char *g, unsigned char *b, int width, int height)
{
    MEASURE_TIME1;
    Image *x = new Image;
    x->width = width;
    x->height = height;
    x->image.channel[0] = r;
    x->image.channel[1] = g;
    x->image.channel[2] = b;
    x->image.channel[3] = 0;
    x->image.pitch[0] = width;
    x->image.pitch[1] = width;
    x->image.pitch[2] = width;
    MEASURE_TIME2("createImage");
    return x;
}

SKR::Image *SKR::jpegde::createImage(unsigned char *r, int width, int height)
{
    MEASURE_TIME1;
    Image *x = new Image;
    x->width = width;
    x->height = height;
    x->image.channel[0] = r;
    x->image.channel[1] = r;
    x->image.channel[2] = r;
    x->image.channel[3] = 0;
    x->image.pitch[0] = width;
    x->image.pitch[1] = width;
    x->image.pitch[2] = width;
    MEASURE_TIME2("createImage");
    return x;
}