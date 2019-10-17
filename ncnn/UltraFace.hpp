//
//  UltraFace.hpp
//  UltraFaceTest
//
//  Created by vealocia on 2019/10/17.
//  Copyright © 2019 vealocia. All rights reserved.
//

#ifndef UltraFace_hpp
#define UltraFace_hpp

#pragma once
#include "include/net.h"
#include "include/gpu.h"
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

#define NMS_HARD 1
#define NMS_SOFT 2
#define NMS_BLEND 3

typedef struct FaceInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    
    float* landmarks;
};

class UltraFace {
public:
    UltraFace(const std::string& bin_path, const std::string& param_path, int input_size, int num_thread_ = 1, int topk_ = -1, float score_threshold_ = 0.9, float iou_threshold_ = 0.1);
    ~UltraFace();
    
    int detect(ncnn::Mat& img, std::vector<FaceInfo>& face_list);
    
private:
    void generateBBox(std::vector<FaceInfo>& bbox_collection, ncnn::Mat scores, ncnn::Mat boxes, float score_threshold, int num_anchors);
    void nms(std::vector<FaceInfo>& input, std::vector<FaceInfo>& output);
private:
    ncnn::Net ultraface;

    int num_thread;
    int image_w;
    int image_h;
    
    int in_w;
    int in_h;
    int num_anchors;
    
    int topk;
    float score_threshold;
    float iou_threshold;
    
    std::string param_file_name;
    std::string bin_file_name;
    
    const float mean_vals[3] = { 127, 127, 127 };
    const float norm_vals[3] = { 1.0/128, 1.0/128, 1.0/128};
};

#endif /* UltraFace_hpp */
