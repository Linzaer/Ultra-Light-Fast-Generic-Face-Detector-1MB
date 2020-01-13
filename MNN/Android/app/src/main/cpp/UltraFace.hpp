//  Created by Linzaer on 2019/11/15.
//  Copyright © 2019 Linzaer. All rights reserved.

#ifndef UltraFace_hpp
#define UltraFace_hpp

#pragma once

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "net.h"

//#include "Bbox.h"

#define num_featuremap 4
#define hard_nms 1
#define blending_nms 2 /* mix nms was been proposaled in paper blaze face, aims to minimize the temporal jitter*/
typedef struct FaceInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;

} FaceInfo;

class UltraFace {
public:
    UltraFace(std::string &mnn_path, int input_width, int input_length, int num_thread_ = 4, float score_threshold_ = 0.7, float iou_threshold_ = 0.35);

    //~UltraFace();

    //int BilinearInterpolationCol(unsigned char * src, unsigned char * des, int srcW, int srcH, int desH);
    int detect(cv::Mat &raw_image, std::vector<FaceInfo> &face_list);
    //float* scores, float* boxes
    void generateBBox(std::vector<FaceInfo> &bbox_collection,  std::vector<float> scores, std::vector<float> boxes);
    //void generateBBox(cv::Mat score, cv::Mat location, std::vector<FaceInfo>& boundingBox_, float scale=0.8);

    void nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, int type = blending_nms);

private:
    Inference_engine ultra_net;
//    std::shared_ptr<MNN::Interpreter> ultraface_interpreter;
//    MNN::Session *ultraface_session = nullptr;
//    MNN::Tensor *input_tensor = nullptr;

    int num_thread;
    int image_w;
    int image_h;

    int in_w;
    int in_h;
    int num_anchors;

    float score_threshold;
    float iou_threshold;

//
    float mean_vals[3] = {127, 127, 127};
    float norm_vals[3] = {1.0 / 128, 1.0 / 128, 1.0 / 128};

//    std::vector<float> mean_vals{ 127.5, 127.5, 127.5 };
//    std::vector<float> norm_vals{ 0.0078125, 0.0078125, 0.0078125 };

    const float center_variance = 0.1;
    const float size_variance = 0.2;
    const std::vector<std::vector<float>> min_boxes = {
            {10.0f,  16.0f,  24.0f},
            {32.0f,  48.0f},
            {64.0f,  96.0f},
            {128.0f, 192.0f, 256.0f}};
    const std::vector<float> strides = {8.0, 16.0, 32.0, 64.0};
    std::vector<std::vector<float>> featuremap_size;
    std::vector<std::vector<float>> shrinkage_size;
    std::vector<int> w_h_list;

    std::vector<std::vector<float>> priors = {};
};

#endif /* UltraFace_hpp */
