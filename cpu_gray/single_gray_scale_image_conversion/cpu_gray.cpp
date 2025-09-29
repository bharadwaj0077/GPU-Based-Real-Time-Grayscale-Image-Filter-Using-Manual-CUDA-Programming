#include <cstdio>
#include <cstdlib>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


void rgb_to_gray_cpu(const unsigned char* rgb, unsigned char* gray,
                     int w, int h, int c) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = (y * w + x) * c;
            float yv = 0.299f * rgb[idx + 0] +
                       0.587f * rgb[idx + 1] +
                       0.114f * rgb[idx + 2];
            gray[y * w + x] = (unsigned char)(yv + 0.5f);
        }
    }
}

int main() {
    const char* in_name  = "input.png";   
    const char* out_name = "gray_cpu.png"; 

    int w = 0, h = 0, c = 0;
    unsigned char* img = stbi_load(in_name, &w, &h, &c, 0);
    if (!img) {
        fprintf(stderr, "Failed to load %s: %s\n", in_name, stbi_failure_reason());
        return 1;
    }
    if (c < 3) {
        fprintf(stderr, "Need RGB image (got %d channels)\n", c);
        stbi_image_free(img);
        return 2;
    }

    size_t out_bytes = (size_t)w * h;
    unsigned char* out = (unsigned char*)malloc(out_bytes);

    
    auto start = std::chrono::high_resolution_clock::now();

    rgb_to_gray_cpu(img, out, w, h, c);

    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    printf("⏱ CPU conversion time: %.3f ms\n", elapsed.count());

    
    stbi_write_png(out_name, w, h, 1, out, w);
    printf("✅ Wrote %s (%dx%d)\n", out_name, w, h);

    stbi_image_free(img);
    free(out);
    return 0;
}
