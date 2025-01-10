#include "./core/headers.cuh"
// ŞÜKRÜ ÇİRİŞ 2024

using namespace SKR;

int main()
{
    std::vector<unsigned char> *encodedjpeg = jpegde::getInstance().readJPEG("./files/package_frames/11.jpg");
    Image *decodedjpeg = jpegde::getInstance().decodeJPEG(*encodedjpeg);

    imageprocess::getInstance().getGray(decodedjpeg);


    delete encodedjpeg;
    jpegde::getInstance().freeJPEG(decodedjpeg);
    return 0;
}