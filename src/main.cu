#include "headers.h"

int main()
{
    std::vector<unsigned char> jpegdecoded;
    std::vector<std::vector<unsigned char>> jpegencoded;
    jpegprocess::getInstance().loadJPEG("./files/arm.jpg", jpegdecoded);
    jpegprocess::getInstance().decodeJPEG(jpegdecoded, jpegencoded);

    return 0;
}