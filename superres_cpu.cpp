#include <opencv2/opencv.hpp>
#include <opencv2/dnn_superres.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace dnn_superres;
using namespace std;
using namespace std::chrono;
using namespace cv::dnn;
int main() {
    string srModel = "models/FSRCNN_x4.pb";
    string modelo = "fsrcnn";
    int escala = 4;

    DnnSuperResImpl sr;
    sr.readModel(srModel);
    sr.setModel(modelo, escala);

    sr.setPreferableBackend(DNN_BACKEND_OPENCV);
    sr.setPreferableTarget(DNN_TARGET_CPU);

    VideoCapture cap("oficina.mp4");
    if (!cap.isOpened()) {
        cout << "No se pudo abrir el video" << endl;
        return -1;
    }

    Mat frame, imgROI, resultado;
    int frameCount = 0;
    double fps = 0.0;
    auto startTime = high_resolution_clock::now();
    Rect roi;

    namedWindow("Original", WINDOW_AUTOSIZE);
    namedWindow("Super Resolución", WINDOW_AUTOSIZE);

    while (cap.read(frame)) {
        if (frame.empty())
            break;

        if (imgROI.empty()) {
            roi.x = (frame.cols / 2) - ((int)(frame.cols / 2) * 0.3);
            roi.y = (frame.rows / 2) - ((int)(frame.rows / 2) * 0.3);
            roi.width = ((int)(frame.cols / 2) * 0.3) * 2;
            roi.height = ((int)(frame.rows / 2) * 0.3) * 2;
        }

        imgROI = frame(roi);
        sr.upsample(imgROI, resultado);

        frameCount++;
        auto currentTime = high_resolution_clock::now();
        double elapsed = duration_cast<duration<double>>(currentTime - startTime).count();
        if (elapsed >= 1.0) {
            fps = frameCount / elapsed;
            frameCount = 0;
            startTime = currentTime;
        }

        putText(resultado, "FPS: " + to_string(fps), Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

        imshow("Original", imgROI);
        imshow("Super Resolución", resultado);

        if (waitKey(1) == 27)
            break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
