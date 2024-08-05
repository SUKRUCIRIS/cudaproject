#include "jpegprocess.h"
#include "utility.h"
// ŞÜKRÜ ÇİRİŞ 2024

jpegprocess::jpegprocess(void)
{
    CHECK_NVJPEG(nvjpegCreateSimple(&handle))
    CHECK_NVJPEG(nvjpegJpegStateCreate(handle, &state))
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_NVJPEG(nvjpegEncoderStateCreate(handle, &enc_state, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsCreate(handle, &enc_params, stream));
}

jpegprocess &jpegprocess::getInstance()
{
    static jpegprocess ins;
    return ins;
}

std::vector<unsigned char> *jpegprocess::readJPEG(const std::string &filename)
{
    std::vector<unsigned char> *buffer = new std::vector<unsigned char>;
    std::cout << "Reading " << filename << std::endl;
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

    return buffer;
}

jpegimage *jpegprocess::decodeJPEG(const std::vector<unsigned char> &jpeg_buffer)
{
    std::cout << "Decoding" << std::endl;
    int nComponents;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];

    jpegimage *output_image = new jpegimage;

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

    return output_image;
}

jpegprocess::~jpegprocess()
{
    CHECK_NVJPEG(nvjpegEncoderParamsDestroy(enc_params));
    CHECK_NVJPEG(nvjpegEncoderStateDestroy(enc_state));
    CHECK_NVJPEG(nvjpegJpegStateDestroy(state));
    CHECK_NVJPEG(nvjpegDestroy(handle));
    CHECK_CUDA(cudaStreamDestroy(stream));
}

void jpegprocess::freeJPEG(jpegimage *image)
{
    for (int i = 0; i < NVJPEG_MAX_COMPONENT; i++)
    {
        CHECK_CUDA(cudaFree(image->image.channel[i]));
    }

    delete image;
}

std::vector<unsigned char> *jpegprocess::encodeJPEG(const jpegimage *image, const int quality, const bool isgray)
{
    std::cout << "Encoding" << std::endl;
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

    return encoded;
}

void jpegprocess::writeJPEG(const std::string &filename, const std::vector<unsigned char> &jpeg_buffer)
{
    std::cout << "Writing " << filename << std::endl;
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
}