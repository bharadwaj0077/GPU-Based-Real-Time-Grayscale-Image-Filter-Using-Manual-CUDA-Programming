#include <cstdio>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

// CUDA kernel: BGR (OpenCV) -> Gray (BT.601)
__global__ void bgr_to_gray(const unsigned char* __restrict__ bgr,
                            unsigned char* __restrict__ gray,
                            int w, int h, int channels, int pitchBytes)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // col
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row
    if (x >= w || y >= h) return;

    const unsigned char* row = bgr + (size_t)y * pitchBytes;
    int idx = x * channels; // OpenCV is interleaved BGRBGR...

    unsigned char B = row[idx + 0];
    unsigned char G = row[idx + 1];
    unsigned char R = row[idx + 2];

    float Y = 0.299f * R + 0.587f * G + 0.114f * B;  // BT.601
    gray[y * w + x] = (unsigned char)(Y + 0.5f);
}

int main(int argc, char** argv)
{
    // Args: input and output (optional)
    const char* in_path  = (argc >= 2) ? argv[1] : "input.mp4";
    const char* out_path = (argc >= 3) ? argv[2] : "output_gray.mp4";

    // Open input video
    cv::VideoCapture cap(in_path);
    if (!cap.isOpened()) {
        std::fprintf(stderr, "Failed to open input video: %s\n", in_path);
        return 1;
    }

    // Get properties
    int width  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0.0) fps = 30.0; // fallback

    // Prepare writer: MP4 (mpeg4) single-channel is not widely supported,
    // so we’ll write as 3-channel BGR with gray replicated to all channels.
    int fourcc = cv::VideoWriter::fourcc('m','p','4','v'); // widely supported
    cv::VideoWriter writer(out_path, fourcc, fps, cv::Size(width, height), /*isColor=*/true);
    if (!writer.isOpened()) {
        std::fprintf(stderr, "Failed to open VideoWriter for: %s\n", out_path);
        return 2;
    }

    // Host buffers
    cv::Mat frame;             // BGR input
    cv::Mat grayCPU(height, width, CV_8UC1);
    cv::Mat grayBGR(height, width, CV_8UC3); // for writing (replicate channels)

    // Device buffers (allocated lazily)
    unsigned char *d_in  = nullptr;
    unsigned char *d_out = nullptr;
    size_t d_in_bytes = 0, d_out_bytes = 0;

    // Timing
    cudaEvent_t t0, t1, t2, t3;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventCreate(&t2);
    cudaEventCreate(&t3);

    double total_ms = 0.0;
    long long frames = 0;

    while (true) {
        if (!cap.read(frame)) break;               // grab next frame
        if (frame.empty()) continue;
        if (frame.channels() != 3) {
            std::fprintf(stderr, "Non-3-channel frame encountered.\n");
            continue;
        }

        const int w = frame.cols;
        const int h = frame.rows;
        const int c = frame.channels();
        const int pitch = (int)frame.step;         // bytes per row
        const size_t inBytes  = (size_t)pitch * h; // full buffer bytes
        const size_t outBytes = (size_t)w * h;     // grayscale bytes

        // (Re)alloc device memory if needed
        if (inBytes != d_in_bytes || outBytes != d_out_bytes) {
            if (d_in)  cudaFree(d_in);
            if (d_out) cudaFree(d_out);
            cudaMalloc(&d_in,  inBytes);
            cudaMalloc(&d_out, outBytes);
            d_in_bytes = inBytes;
            d_out_bytes = outBytes;
        }

        cudaEventRecord(t0);

        // H2D copy (whole frame buffer)
        cudaMemcpy(d_in, frame.data, inBytes, cudaMemcpyHostToDevice);
        cudaEventRecord(t1);

        // Launch kernel
        dim3 block(32, 16);
        dim3 grid((w + block.x - 1) / block.x,
                  (h + block.y - 1) / block.y);
        bgr_to_gray<<<grid, block>>>(d_in, d_out, w, h, c, pitch);
        cudaDeviceSynchronize();
        cudaEventRecord(t2);

        // D2H copy (grayscale)
        cudaMemcpy(grayCPU.data, d_out, outBytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(t3);
        cudaEventSynchronize(t3);

        float ms_h2d=0, ms_k=0, ms_d2h=0, ms_total=0;
        cudaEventElapsedTime(&ms_h2d, t0, t1);
        cudaEventElapsedTime(&ms_k,   t1, t2);
        cudaEventElapsedTime(&ms_d2h, t2, t3);
        cudaEventElapsedTime(&ms_total,t0, t3);

        total_ms += ms_total;
        frames++;

        // For writing: replicate gray → BGR
        cv::cvtColor(grayCPU, grayBGR, cv::COLOR_GRAY2BGR);
        writer.write(grayBGR);

        // Optional on-screen preview (press 'q' to quit)
        cv::Mat preview;
        cv::hconcat(frame, grayBGR, preview);
        char info[128];
        double fps_inst = 1000.0 / ms_total;
        std::snprintf(info, sizeof(info), "H2D %.2f ms | K %.2f ms | D2H %.2f ms | %.2f ms (%.1f FPS)",
                      ms_h2d, ms_k, ms_d2h, ms_total, fps_inst);
        cv::putText(preview, info, {20, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0,255,0}, 2);
        cv::imshow("CUDA Video Grayscale (q=quit)", preview);
        char k = (char)cv::waitKey(1);
        if (k == 'q' || k == 27) break;
    }

    if (frames > 0) {
        double avg_ms = total_ms / frames;
        double avg_fps = 1000.0 / avg_ms;
        std::printf("Processed %lld frames | avg %.3f ms/frame (%.1f FPS)\n", frames, avg_ms, avg_fps);
        std::printf("Output saved to: %s\n", out_path);
    }

    // Cleanup
    if (d_in)  cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaEventDestroy(t2); cudaEventDestroy(t3);
    cap.release();
    writer.release();
    cv::destroyAllWindows();
    return 0;
}
