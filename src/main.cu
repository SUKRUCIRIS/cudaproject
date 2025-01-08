#include "./core/headers.cuh"
// ŞÜKRÜ ÇİRİŞ 2024

using namespace SKR;

int main()
{
    std::vector<unsigned char> *encodedjpeg = jpegde::getInstance().readJPEG("./files/arm.jpg");
    Image *decodedjpeg = jpegde::getInstance().decodeJPEG(*encodedjpeg);

    imageprocess::getInstance().getGray(decodedjpeg);

    imageprocess::getInstance().getSobelEdges(decodedjpeg, 90);

    std::vector<unsigned char> *outputjpeg = jpegde::getInstance().encodeJPEG(decodedjpeg, 50, false);
    jpegde::getInstance().writeJPEG("./out.jpg", *outputjpeg);
    delete encodedjpeg;
    delete outputjpeg;
    jpegde::getInstance().freeJPEG(decodedjpeg);
    return 0;
}