#include "./core/headers.cuh"
// ŞÜKRÜ ÇİRİŞ 2024

using namespace SKR;

int main()
{
    // target object images, template matcher will average the ssim values of these images
    // all target objects must be in the same size, template matcher will travers on frame with that size
    std::string targetfilenames[7] = {
        "./files/target_object/1.jpg",
        "./files/target_object/2.jpg",
        "./files/target_object/3.jpg",
        "./files/target_object/4.jpg",
        "./files/target_object/5.jpg",
        "./files/target_object/6.jpg",
        "./files/target_object/7.jpg",
    };

    TemplateMatcherSSIMonEdge *tm = new TemplateMatcherSSIMonEdge(targetfilenames, 7);

    // detect the object in the frame
    vec2i result = tm->detectObjectBatch("./files/package_frames/3.jpg");

    // prints top left corner of the detected object
    std::cout << "Result: " << result.x << " " << result.y << std::endl;

    delete tm;

    return 0;
}