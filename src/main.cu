#include "headers.h"
// ŞÜKRÜ ÇİRİŞ 2024

int main()
{
    std::vector<unsigned char> *encodedjpeg = jpegprocess::getInstance().readJPEG("./files/arm.jpg");
    jpegimage *decodedjpeg = jpegprocess::getInstance().decodeJPEG(*encodedjpeg);
    std::vector<unsigned char> *outputjpeg = jpegprocess::getInstance().encodeJPEG(decodedjpeg, 50, true);
    jpegprocess::getInstance().writeJPEG("./out.jpg", *outputjpeg);

    delete encodedjpeg;
    delete outputjpeg;
    jpegprocess::getInstance().freeJPEG(decodedjpeg);
    return 0;
}