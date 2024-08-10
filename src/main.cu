#include "./core/headers.h"
// ŞÜKRÜ ÇİRİŞ 2024

using namespace SKR;

int main()
{
    std::vector<unsigned char> *encodedjpeg = jpegprocess::getInstance().readJPEG("./files/manzara.jpg");
    jpegimage *decodedjpeg = jpegprocess::getInstance().decodeJPEG(*encodedjpeg);

    imageprocess::getInstance().getSmooth(decodedjpeg);

    std::vector<unsigned char> *outputjpeg = jpegprocess::getInstance().encodeJPEG(decodedjpeg, 50, false);
    jpegprocess::getInstance().writeJPEG("./out.jpg", *outputjpeg);
    delete encodedjpeg;
    delete outputjpeg;
    jpegprocess::getInstance().freeJPEG(decodedjpeg);
    return 0;
}