//
//  main.cpp
//  UltraFaceTest
//
//  Created by vealocia on 2019/10/17.
//  Copyright © 2019 vealocia. All rights reserved.
//

#include <opencv4/opencv2/opencv.hpp> // opencv library path should be adjustment on your own work place
#include <iostream>
#include "UltraFace.hpp"

int main(){
    std::string bin_path="source/ncnn.bin";
    std::string param_path="source/ncnn.param";
    std::string image_file="source/test.jpg";
    
    UltraFace ultraface(bin_path, param_path, 320); // config model input
    cv::Mat frame = cv::imread(image_file);
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    ncnn::Mat inmat = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_RGB, frame.cols, frame.rows);
    
    std::vector<FaceInfo> face_info;
    ultraface.detect(inmat, face_info);

    for (int i = 0; i < face_info.size(); i++)
    {
        auto face = face_info[i];
        cv::Point pt1(face.x1, face.y1);
        cv::Point pt2(face.x2, face.y2);
        cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);
    }

    cv::namedWindow("UltraFace");
    cv::imshow("UltraFace", frame);
    cv::waitKey();
    cv::imwrite("result.jpg", frame);
    return 0;
}
