#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <chrono>
#include <iostream>

void processGPU(const cv::Mat& frame) {

    auto total_start = std::chrono::high_resolution_clock::now();

    // Medimos transferencia CPU → GPU
    auto t1 = std::chrono::high_resolution_clock::now();
    cv::cuda::GpuMat d_frame(frame);
    auto t2 = std::chrono::high_resolution_clock::now();

    cv::cuda::GpuMat d_gray, d_blur, d_morph1, d_morph2, d_edges, d_eq;

    // Procesamiento GPU
    auto start_proc = std::chrono::high_resolution_clock::now();

    cv::cuda::cvtColor(d_frame, d_gray, cv::COLOR_BGR2GRAY);

    static auto gauss = cv::cuda::createGaussianFilter(d_gray.type(), d_gray.type(), cv::Size(5, 5), 1.5);
    gauss->apply(d_gray, d_blur);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    static auto dilate = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, d_blur.type(), kernel);
    static auto erode = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, d_blur.type(), kernel);
    dilate->apply(d_blur, d_morph1);
    erode->apply(d_morph1, d_morph2);

    static auto canny = cv::cuda::createCannyEdgeDetector(50, 150);
    canny->detect(d_morph2, d_edges);

    d_edges.convertTo(d_edges, CV_8UC1);
    cv::cuda::equalizeHist(d_edges, d_eq);

    auto end_proc = std::chrono::high_resolution_clock::now();

    // Transferencia GPU → CPU
    auto t3 = std::chrono::high_resolution_clock::now();
    cv::Mat result;
    d_eq.download(result);
    auto t4 = std::chrono::high_resolution_clock::now();

    auto total_end = std::chrono::high_resolution_clock::now();

    // Reporte de tiempos
    std::cout << "[Transferencia CPU → GPU] " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;

    std::cout << "[Procesamiento GPU] " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_proc - start_proc).count() << " ms" << std::endl;

    std::cout << "[Transferencia GPU → CPU] " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << " ms" << std::endl;

    std::cout << "[Total incluyendo transferencias] " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count() << " ms" << std::endl;

    cv::imshow("Resultado GPU", result);
}

int main() {
    cv::Mat frame = cv::imread("elefante.jpg");
    if (frame.empty()) {
        std::cerr << "❌ Error al cargar la imagen." << std::endl;
        return -1;
    }

    std::cout << "Tamaño de la imagen: " << frame.cols << " x " << frame.rows << std::endl;
    std::cout << "Canales: " << frame.channels() << std::endl;

    processGPU(frame);

    cv::waitKey(0);
    return 0;
}