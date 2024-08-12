#include "./core/headers.h"
// ŞÜKRÜ ÇİRİŞ 2024

using namespace SKR;

int main()
{
    std::vector<unsigned char> *encodedjpeg = jpegprocess::getInstance().readJPEG("./files/arm.jpg");
    jpegimage *decodedjpeg = jpegprocess::getInstance().decodeJPEG(*encodedjpeg);

    imageprocess::getInstance().getGray(decodedjpeg);

    imageprocess::getInstance().getSobelEdges(decodedjpeg, 75);

    std::vector<unsigned char> *outputjpeg = jpegprocess::getInstance().encodeJPEG(decodedjpeg, 50, false);
    jpegprocess::getInstance().writeJPEG("./out.jpg", *outputjpeg);
    delete encodedjpeg;
    delete outputjpeg;
    jpegprocess::getInstance().freeJPEG(decodedjpeg);
    return 0;
}