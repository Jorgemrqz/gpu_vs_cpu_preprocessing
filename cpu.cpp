#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

void processCPU(const cv::Mat& frame) {
    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat gray, blur, morph, edges, hist_eq;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 1.5);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(blur, morph, kernel);
    cv::erode(morph, morph, kernel);

    cv::Canny(morph, edges, 50, 150);
    cv::equalizeHist(edges, hist_eq);

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "[CPU] Tiempo: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
              << " ms" << std::endl;

    cv::imshow("Resultado CPU", hist_eq);
}

int main() {
    cv::Mat frame = cv::imread("elefante.jpg");
    if (frame.empty()) {
        std::cerr << "❌ Error al cargar la imagen." << std::endl;
        return -1;
    }

    std::cout << "Tamaño de la imagen: " << frame.cols << " x " << frame.rows << std::endl;
    std::cout << "Canales: " << frame.channels() << std::endl;

    processCPU(frame);

    cv::waitKey(0);
    return 0;
}