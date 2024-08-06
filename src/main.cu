#include "./core/headers.h"
// ŞÜKRÜ ÇİRİŞ 2024

int main()
{
    std::vector<unsigned char> *encodedjpeg = jpegprocess::getInstance().readJPEG("./files/arm.jpg");
    jpegimage *decodedjpeg = jpegprocess::getInstance().decodeJPEG(*encodedjpeg);

    int blockn = (decodedjpeg->width * decodedjpeg->height + 511) / 512;
    getHighContrast<<<blockn, 512>>>(decodedjpeg->image.channel[0], decodedjpeg->image.channel[1],
                                decodedjpeg->image.channel[2], 2, decodedjpeg->width * decodedjpeg->height);

    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<unsigned char> *outputjpeg = jpegprocess::getInstance().encodeJPEG(decodedjpeg, 50, false);
    jpegprocess::getInstance().writeJPEG("./out.jpg", *outputjpeg);
    delete encodedjpeg;
    delete outputjpeg;
    jpegprocess::getInstance().freeJPEG(decodedjpeg);
    return 0;
}