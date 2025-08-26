#include <cstdio>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void rgb_to_gray(const unsigned char* rgb, unsigned char* gray,
                            int w, int h, int c) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = (y*w + x)*c;
    float yv = 0.299f*rgb[idx+0] + 0.587f*rgb[idx+1] + 0.114f*rgb[idx+2];
    gray[y*w + x] = (unsigned char)(yv + 0.5f);
}

int main() {
    const char* in_name  = "input.png";   // <- fixed, no ambiguity
    const char* out_name = "gray.png";

    int w=0, h=0, c=0;
    unsigned char* img = stbi_load(in_name, &w, &h, &c, 0);
    if (!img) {
        fprintf(stderr, "Failed to load %s: %s\n", in_name, stbi_failure_reason());
        return 1;
    }
    if (c < 3) { fprintf(stderr, "Need RGB image\n"); return 2; }

    size_t in_bytes = (size_t)w*h*c, out_bytes = (size_t)w*h;
    unsigned char *d_in=nullptr, *d_out=nullptr;
    cudaMalloc(&d_in, in_bytes);
    cudaMalloc(&d_out, out_bytes);
    cudaMemcpy(d_in, img, in_bytes, cudaMemcpyHostToDevice);

    dim3 block(32,16), grid((w+block.x-1)/block.x, (h+block.y-1)/block.y);
    rgb_to_gray<<<grid, block>>>(d_in, d_out, w, h, c);
    cudaDeviceSynchronize();

    unsigned char* out = (unsigned char*)malloc(out_bytes);
    cudaMemcpy(out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    stbi_write_png(out_name, w, h, 1, out, w);
    printf("✅ Wrote %s (%dx%d)\n", out_name, w, h);

    cudaFree(d_in); cudaFree(d_out);
    stbi_image_free(img); free(out);
    return 0;
}
