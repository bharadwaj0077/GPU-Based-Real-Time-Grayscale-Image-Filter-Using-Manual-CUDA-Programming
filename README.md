# GPU-Based-Real-Time-Grayscale-Image-Filter-Using-Manual-CUDA-Programming

Project HPC- Team B12
The first part of the project demonstrates converting a single color image to grayscale using *CUDA parallel processing*. Each GPU thread handles one pixel, making the process massively parallel compared to a CPU loop.

---

## 1. Prerequisites

### Hardware
- NVIDIA GPU (*GeForce RTX 4070* or similar)

### Software
- *Windows 11* (x64)
- *NVIDIA GPU Driver* (verified with nvidia-smi)
- *Visual Studio 2022 (Community Edition)*
  - Workload: Desktop development with C++
- *CUDA Toolkit 12.9 / 13.0*
  - Verified with:
    bat
    nvcc --version
    
- *STB Image Libraries* (header-only, no install)
  - stb_image.h (for loading images)
  - stb_image_write.h (for saving images)

---

## 2. Folder Structure
C:\Users\srima\Desktop\HPC_test\CudaGrayscale
-
CudaGrayscale/
  - grey_scale.cu          
  - stb_image.h            
  - stb_image_write.h      
  - input.jpg              
  - gray.exe              
  - grey.png            

After running the program, a new file will be generated that is grey.png:
grey.png contain the greyscale convertion of input image.

# CPU-Based Sequential Grayscale Image Filter

## 1. Prerequisites

# Hardware
* Any modern CPU (e.g., Intel Core i7, AMD Ryzen 5, or similar)

* Sufficient RAM for image loading and processing.

# Software
* Windows 11 (x64)

* Visual Studio 2022 (Community Edition)

* Workload: Desktop development with C++

* C++ Compiler (specifically g++)

* C++17 standard is recommended for file system features.

* STB Image Libraries (header-only, no install)

* stb_image.h (for loading images)

* stb_image_write.h (for saving images)


# GPU-Based-Grayscale-Video-Filter-CUDA-Programming
---

## 1. Prerequisites

### Hardware
- NVIDIA GPU (*GeForce RTX 4070* or similar)

### Software
- *Windows 11* (x64)
- *NVIDIA GPU Driver* (verified with `nvidia-smi`)
- *Visual Studio 2022 (Community Edition)*
  - Workload: Desktop development with C++
- *CUDA Toolkit 12.9 / 13.0*
  - Verified with:
    ```bash
    nvcc --version
    ```
- *STB Image Libraries* (header-only, no install)
  - stb_image.h (for loading images)
  - stb_image_write.h (for saving images)

---

## 2. Folder Structure
C:\Users\srima\Desktop\HPC_test\Cuda_grayscale_video
-
Root\
  - input.mp4
  - output_gray.mp4
  - video_gray.exe
  - video_gray.exp
  - video_gray.lib
  - video_grayscale.cu

After running the program, a new file will be generated — **output_gray.mp4**.  
It contains the grayscale conversion of the input video.
---
# Web Integration — Full Project Description

This extended phase integrates both *CPU* and *CUDA* executables into a *web-based system* with a *Node.js + Express backend*.  
Users can upload images or videos through a browser and choose between *CPU (C++)* or *GPU (CUDA)* for real-time grayscale conversion.
---
## Folder Structure
-
C:\Root
- public/
    - index.html
- bin/
    - cpu_gray_image.exe
    - cuda_gray_image.exe
    - cpu_gray_video.exe
    - cuda_gray_video.exe
- uploads/ # Auto-created during runtime
- results/ # Processed outputs saved here 
- server.js # Node.js backend
- package.json

---

## How It Works

1. User opens the web app at http://localhost:3000.
2. Chooses:
   - *Image* or *Video*
   - *CPU* or *GPU*
3. Uploads a file → Backend stores it in /uploads/.
4. The backend runs the appropriate executable from /bin/.
5. The processed output is saved to /results/ and displayed in the UI.

---

## Node.js Setup

```bash
npm init -y
npm i express multer cors
node server.js

Now in brower open http://localhost:3000
There you can input the image or video and select the cpu or gpu it will run the corresponding and give the grayscale output.
```
## Conclusion

This project successfully demonstrates real-time image and video processing using CUDA parallel programming.
It integrates high-performance native computing with a modern web interface, showcasing the advantage of GPU acceleration over CPU-based execution.