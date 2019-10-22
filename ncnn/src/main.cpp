//
//  main.cpp
//  UltraFaceTest
//
//  Created by vealocia on 2019/10/17.
//  Copyright © 2019 vealocia. All rights reserved.
//

#include "UltraFace.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
    if (argc <= 3) {
        fprintf(stderr, "Usage: %s <ncnn bin> <ncnn param> [image files...]\n", argv[0]);
        return 1;
    }

    std::string bin_path = argv[1];
    std::string param_path = argv[2];
    cv::namedWindow("UltraFace");
    UltraFace ultraface(bin_path, param_path, 320); // config model input

    for (int i = 3; i < argc; i++) {
        std::string image_file = argv[i];
        std::cout << "Processing " << image_file << std::endl;

        cv::Mat frame = cv::imread(image_file);
        ncnn::Mat inmat = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);

        std::vector<FaceInfo> face_info;
        ultraface.detect(inmat, face_info);

        for (int i = 0; i < face_info.size(); i++) {
            auto face = face_info[i];
            cv::Point pt1(face.x1, face.y1);
            cv::Point pt2(face.x2, face.y2);
            cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("UltraFace", frame);
        cv::waitKey();
        cv::imwrite("result.jpg", frame);
    }
    return 0;
}
