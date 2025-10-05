
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Declare external functions
extern "C" unsigned char* load_image(const char* filename, int* w, int* h, int* c);
extern "C" int save_gray_image(const char* filename, unsigned char* data, int w, int h);
extern "C" void free_image(unsigned char* data);

__global__ void rgb2gray(const unsigned char* rgb, unsigned char* gray, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx < total) {
        int r = rgb[3 * idx + 0];
        int g = rgb[3 * idx + 1];
        int b = rgb[3 * idx + 2];
        gray[idx] = static_cast<unsigned char>(0.299f*r + 0.587f*g + 0.114f*b);
    }
}

int main(int argc, char** argv) {
    if(argc < 3) {
        printf("Usage: %s input_image output_image\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    unsigned char* h_rgb = load_image(argv[1], &width, &height, &channels);
    if(!h_rgb) {
        printf("Failed to load image %s\n", argv[1]);
        return -1;
    }
    
    size_t rgbBytes = width * height * 3;
    size_t grayBytes = width * height;

    unsigned char* h_gray = (unsigned char*)malloc(grayBytes);

    unsigned char *d_rgb, *d_gray;
    cudaMalloc(&d_rgb, rgbBytes);
    cudaMalloc(&d_gray, grayBytes);

    cudaMemcpy(d_rgb, h_rgb, rgbBytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;
    int blocks = (width * height + threads - 1) / threads;

    cudaEventRecord(start);
    rgb2gray<<<blocks, threads>>>(d_rgb, d_gray, width, height);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    printf("Processed image %dx%d in %.3f ms\n", width, height, elapsed);

    cudaMemcpy(h_gray, d_gray, grayBytes, cudaMemcpyDeviceToHost);

    if(!save_gray_image(argv[2], h_gray, width, height)) {
        printf("Failed to save image %s\n", argv[2]);
    } else {
        printf("Saved grayscale image to %s\n", argv[2]);
    }

    cudaFree(d_rgb);
    cudaFree(d_gray);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free_image(h_rgb);
    free(h_gray);

    return 0;
}