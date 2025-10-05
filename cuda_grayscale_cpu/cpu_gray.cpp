#include "stb_image.h"
#include "stb_image_write.h"

void cpuGray(const unsigned char* rgb, unsigned char* gray, int width, int height) {
    int totalPixels = width * height;
    for (int i = 0; i < totalPixels; i++) {
        int r = rgb[3*i + 0];
        int g = rgb[3*i + 1];
        int b = rgb[3*i + 2];
        gray[i] = static_cast<unsigned char>(0.299*r + 0.587*g + 0.114*b);
    }
}
