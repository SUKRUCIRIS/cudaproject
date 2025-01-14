#include "./core/headers.cuh"
// ŞÜKRÜ ÇİRİŞ 2024

using namespace SKR;

int main()
{
    std::string targetfilenames[7] = {
        "./files/target_object/1.jpg",
        "./files/target_object/2.jpg",
        "./files/target_object/3.jpg",
        "./files/target_object/4.jpg",
        "./files/target_object/5.jpg",
        "./files/target_object/6.jpg",
        "./files/target_object/7.jpg",
    };

    TemplateMatcher *tm = new TemplateMatcher(targetfilenames, 7);

    vec2i result = tm->detectObjectBatch("./files/package_frames/3.jpg");

    std::cout << "Result: " << result.x << " " << result.y << std::endl;

    delete tm;

    return 0;
}