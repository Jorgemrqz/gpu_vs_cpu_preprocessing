#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>

using namespace cv;
using namespace dnn;
using namespace std;

int main() {
    // Rutas de los archivos
    string modelWeights = "models/frozen_inference_graph.pb";
    string modelConfig = "models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt";
    string classesFile = "models/labels.txt";

    // Cargar las clases
    vector<string> classes;
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) {
        classes.push_back(line);
    }

    // Cargar la red
    Net net = readNetFromTensorflow(modelWeights, modelConfig);

    // Usar CUDA si está disponible
    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(DNN_TARGET_CUDA); 

    // Leer imagen de prueba
    Mat frame = imread("images.jpeg");
    if (frame.empty()) {
        cout << "No se pudo cargar la imagen" << endl;
        return -1;
    }

    // Preprocesar la imagen (formato estándar para este modelo)
    Mat blob = blobFromImage(frame, 1.0 / 127.5, Size(320, 320), Scalar(127.5, 127.5, 127.5), true, false);

    // Inferencia
    net.setInput(blob);
    Mat detections = net.forward();

    // Procesar los resultados
    Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());

    for (int i = 0; i < detectionMat.rows; i++) {
        float confidence = detectionMat.at<float>(i, 2);
        if (confidence > 0.5) {
            int classId = static_cast<int>(detectionMat.at<float>(i, 1)); // ID de la clase
            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

            rectangle(frame, Point(xLeftBottom, yLeftBottom), Point(xRightTop, yRightTop), Scalar(0, 255, 0), 2);

            // Dibuja la etiqueta si el índice es válido
            if (classId > 0 && classId <= classes.size()) {
                string label = format("%s: %.2f", classes[classId - 1].c_str(), confidence);
                int baseLine = 0;
                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                yLeftBottom = max(yLeftBottom, labelSize.height);
                rectangle(frame, Point(xLeftBottom, yLeftBottom - labelSize.height),
                          Point(xLeftBottom + labelSize.width, yLeftBottom + baseLine),
                          Scalar(255, 255, 255), FILLED);

                putText(frame, label, Point(xLeftBottom, yLeftBottom),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
            }

            cout << "Detección: " << classes[classId - 1] << " con confianza: " << confidence << endl;
        }
    }

    imshow("Detecciones", frame);
    waitKey(0);

    return 0;
}
