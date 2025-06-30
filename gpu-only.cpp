#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <chrono>
#include <iostream>

int main() {
    cv::Mat frame = cv::imread("elefante.jpg");
    if (frame.empty()) {
        std::cerr << "❌ Error al cargar la imagen." << std::endl;
        return -1;
    }

    std::cout << "Tamaño de la imagen: " << frame.cols << " x " << frame.rows << std::endl;

    cv::cuda::GpuMat d_frame, d_gray, d_blur, d_morph, d_edges;
    cv::Mat result;

    // Crear filtros una sola vez
    auto gauss = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(5, 5), 1.5);
    auto dilate = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    auto erode = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    auto canny = cv::cuda::createCannyEdgeDetector(50, 150);

    // Medir transferencia CPU → GPU
    auto start_upload = std::chrono::high_resolution_clock::now();
    d_frame.upload(frame);
    auto end_upload = std::chrono::high_resolution_clock::now();
    std::cout << "[Transferencia CPU → GPU] "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_upload - start_upload).count()
              << " ms" << std::endl;

    // Medición GPU-only
    auto start_gpu = std::chrono::high_resolution_clock::now();
    cv::cuda::cvtColor(d_frame, d_gray, cv::COLOR_BGR2GRAY);
    gauss->apply(d_gray, d_blur);
    dilate->apply(d_blur, d_morph);
    erode->apply(d_morph, d_morph);
    canny->detect(d_morph, d_edges);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "[GPU-only] Tiempo procesamiento: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()
              << " ms" << std::endl;

    // Medir descarga GPU → CPU
    auto start_download = std::chrono::high_resolution_clock::now();
    d_edges.download(result);
    auto end_download = std::chrono::high_resolution_clock::now();
    std::cout << "[Transferencia GPU → CPU] "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_download - start_download).count()
              << " ms" << std::endl;

    cv::imshow("Resultado GPU", result);
    cv::waitKey(0);
    return 0;
}
