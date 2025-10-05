// #define STB_IMAGE_IMPLEMENTATION
// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #include "stb_image.h"
// #include "stb_image_write.h"

// extern "C" unsigned char* load_image(const char* filename, int* w, int* h) {
//     int c;
//     return stbi_load(filename, w, h, &c, 3);
// }

// extern "C" void save_gray_image(const char* filename, unsigned char* data, int w, int h) {
//     stbi_write_jpg(filename, w, h, 1, data, 100);
// }

// extern "C" void free_image(unsigned char* data) {
//     stbi_image_free(data);
// }

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

extern "C" {
    unsigned char* load_image(const char* filename, int* w, int* h, int* c) {
        return stbi_load(filename, w, h, c, 3);
    }

    int save_gray_image(const char* filename, unsigned char* data, int w, int h) {
        return stbi_write_jpg(filename, w, h, 1, data, 100);
    }

    void free_image(unsigned char* data) {
        stbi_image_free(data);
    }
}