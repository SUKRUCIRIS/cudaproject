#include "jpegprocess.h"
#include "utility.h"

jpegprocess::jpegprocess(void)
{
    CHECK_NVJPEG(nvjpegCreateSimple(&handle))
    CHECK_NVJPEG(nvjpegJpegStateCreate(handle, &state))
    CHECK_CUDA(cudaStreamCreate(&stream));
}

jpegprocess &jpegprocess::getInstance()
{
    static jpegprocess ins;
    return ins;
}

bool jpegprocess::loadJPEG(const std::string &filename, std::vector<unsigned char> &buffer)
{
    std::cout << "Loading " << filename << std::endl;
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    buffer.resize(size);
    if (!file.read(reinterpret_cast<char *>(buffer.data()), size))
    {
        std::cerr << "Failed to read the file: " << filename << std::endl;
        return false;
    }

    return true;
}

void jpegprocess::decodeJPEG(const std::vector<unsigned char> &jpeg_buffer, std::vector<std::vector<unsigned char>> &output_buffers)
{
    int nComponents;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];

    CHECK_NVJPEG(nvjpegGetImageInfo(handle, jpeg_buffer.data(), jpeg_buffer.size(), &nComponents, &subsampling, widths, heights));

    output_buffers.resize(nComponents);

    nvjpegImage_t output_image = {0};

    for (int i = 0; i < nComponents; i++)
    {
        std::cout << "Component: " << i << " Width: " << widths[i] << " Height: " << heights[i] << std::endl;
        int channel_size = heights[i] * widths[i];
        output_buffers[i].resize(channel_size);
        output_image.channel[i] = output_buffers[i].data();
        output_image.pitch[i] = widths[i];
    }

    CHECK_NVJPEG(nvjpegDecode(handle, state, jpeg_buffer.data(), jpeg_buffer.size(), NVJPEG_OUTPUT_RGB, &output_image, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));
}

jpegprocess::~jpegprocess()
{
    CHECK_NVJPEG(nvjpegJpegStateDestroy(state));
    CHECK_NVJPEG(nvjpegDestroy(handle));
    CHECK_CUDA(cudaStreamDestroy(stream));
}