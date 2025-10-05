// #define STB_IMAGE_IMPLEMENTATION
// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #include <iostream>
// #include <chrono>
// #include "stb_image.h"
// #include "stb_image_write.h"

// void cpuGray(const unsigned char* rgb, unsigned char* gray, int width, int height);

// int main(int argc, char** argv) {
//     if (argc < 3) {
//         std::cout << "Usage: ./cpu_gray <input> <output>\n";
//         return 1;
//     }

//     int width, height, channels;
//     unsigned char* img = stbi_load(argv[1], &width, &height, &channels, 3);
//     if (!img) {
//         std::cerr << "Failed to load image!\n";
//         return -1;
//     }

//     unsigned char* gray = new unsigned char[width * height];

//     auto start = std::chrono::high_resolution_clock::now();
//     cpuGray(img, gray, width, height);
//     auto end = std::chrono::high_resolution_clock::now();

//     double duration = std::chrono::duration<double, std::milli>(end - start).count();
//     std::cout << "Image: " << argv[1] << " | CPU Grayscale Time: " << duration << " ms\n";

//     stbi_write_jpg(argv[2], width, height, 1, gray, 100);

//     stbi_image_free(img);
//     delete[] gray;
//     return 0;
// }
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

using namespace std;

void cpuGray(const unsigned char* rgb, unsigned char* gray, int width, int height) {
    int totalPixels = width * height;
    for (int i = 0; i < totalPixels; i++) {
        int r = rgb[3*i + 0];
        int g = rgb[3*i + 1];
        int b = rgb[3*i + 2];
        gray[i] = static_cast<unsigned char>(0.299*r + 0.587*g + 0.114*b);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: cpu_gray.exe <input> <output>\n";
        return 1;
    }

    string inputPath  = argv[1];
    string outputPath = argv[2];

    int width, height, channels;
    unsigned char* img = stbi_load(inputPath.c_str(), &width, &height, &channels, 3);
    if (!img) {
        cerr << "Failed to load image: " << inputPath << endl;
        return -1;
    }

    unsigned char* gray = new unsigned char[width * height];

    auto start = chrono::high_resolution_clock::now();
    cpuGray(img, gray, width, height);
    auto end = chrono::high_resolution_clock::now();

    double duration = chrono::duration<double, milli>(end - start).count();

    cout << "Image: " << inputPath
         << " | Size: " << width << "x" << height
         << " | CPU Grayscale Time: " << duration << " ms\n";

    stbi_write_jpg(outputPath.c_str(), width, height, 1, gray, 100);

    // --- CSV logging without <filesystem> ---
    bool fileExists = false;
    ifstream test("benchmark.csv");
    if (test.good()) fileExists = true;
    test.close();

    ofstream csv("benchmark.csv", ios::app);
    if (!fileExists)
        csv << "Image,Width,Height,CPU_Time(ms)\n";
    csv << inputPath << "," << width << "," << height << "," << duration << "\n";
    csv.close();

    stbi_image_free(img);
    delete[] gray;
    return 0;
}
