//
//  UltraFace.cpp
//  UltraFaceTest
//
//  Created by vealocia on 2019/10/17.
//  Copyright © 2019 vealocia. All rights reserved.
//

#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

#include "UltraFace.hpp"
#include "include/mat.h"

UltraFace::UltraFace(const std::string& bin_path, const std::string& param_path, int input_size, int num_thread_, int topk_, float score_threshold_, float iou_threshold_){
    num_thread = num_thread_;
    topk = topk_;
    score_threshold = score_threshold_;
    iou_threshold = iou_threshold_;
    
    switch(input_size){
        case 128:{
            in_w = 128;
            in_h = 96;
            num_anchors = 708;
            break;
        }
        case 320:{
            in_w = 320;
            in_h = 240;
            num_anchors = 4402;
            break;
        }
        default:{
            printf("unknown input size.");
            exit(-1);
        }
    }
    ultraface.load_param(param_path.data());
    ultraface.load_model(bin_path.data());
}

UltraFace::~UltraFace()
{
    ultraface.clear();
}

int UltraFace::detect(ncnn::Mat& img, std::vector<FaceInfo>& face_list){
    if (img.empty()) {
        std::cout << "image is empty ,please check!" << std::endl;
        return -1;
    }

    image_h = img.h;
    image_w = img.w;

    ncnn::Mat in;
    in = img;
    ncnn::resize_bilinear(img,in, in_w, in_h);
    
    ncnn::Mat ncnn_img = in;
    ncnn_img.substract_mean_normalize(mean_vals, norm_vals);
    
    std::vector<FaceInfo> bbox_collection;
    std::vector<FaceInfo> valid_input;
    
    ncnn::Extractor ex = ultraface.create_extractor();
    ex.set_num_threads(num_thread);
    ex.input("input", ncnn_img);
    
    ncnn::Mat scores;
    ncnn::Mat boxes;
    ex.extract("scores", scores);
    ex.extract("boxes", boxes);
    
    generateBBox(bbox_collection, scores, boxes, score_threshold, num_anchors);
    
    nms(bbox_collection, face_list);
    printf("face num: %d\n", face_list.size());
    
    for(int i=0;i<face_list.size();i++){
        float w,h,maxSize;
        float cenx,ceny;
        w=face_list[i].x2-face_list[i].x1;
        h=face_list[i].y2-face_list[i].y1;

        maxSize = w > h ? w : h;
        cenx=face_list[i].x1+w/2;
        ceny=face_list[i].y1+h/2;
        face_list[i].x1=cenx-maxSize/2>0 ? cenx - maxSize / 2 : 0;
        face_list[i].y1=ceny-maxSize/2>0 ? ceny - maxSize / 2 : 0;
        face_list[i].x2=cenx+maxSize/2>image_w ? image_w-1 : cenx + maxSize / 2;
        face_list[i].y2=ceny+maxSize/2> image_h ? image_h-1 : ceny + maxSize / 2;
    }
    return 0;
}

void UltraFace::generateBBox(std::vector<FaceInfo>& bbox_collection, ncnn::Mat scores, ncnn::Mat boxes, float score_threshold, int num_anchors) {
    for(int i=0; i<scores.h; i++){
        if(scores.channel(0)[i*2+1] > score_threshold){
            FaceInfo rects;
            
            rects.x1 = clip(boxes.channel(0)[i*2] - boxes.channel(0)[(i+num_anchors)*2] / 2.0, 1) * image_w;
            rects.y1 = clip(boxes.channel(0)[i*2+1] - boxes.channel(0)[(i+num_anchors)*2+1] / 2.0, 1) * image_h;
            rects.x2 = clip(boxes.channel(0)[i*2] + boxes.channel(0)[(i+num_anchors)*2] / 2.0, 1) * image_w;
            rects.y2 = clip(boxes.channel(0)[i*2+1] + boxes.channel(0)[(i+num_anchors)*2+1] / 2.0, 1) * image_h;
            rects.score = clip(scores.channel(0)[i*2+1], 1);
            
            bbox_collection.push_back(rects);
        }
    }
}

void UltraFace::nms(std::vector<FaceInfo>& input, std::vector<FaceInfo>& output){
    std::sort(input.begin(), input.end(),
              [](const FaceInfo& a, const FaceInfo& b)
              {
                  return a.score > b.score;
              });

    int box_num = input.size();

    std::vector<int> merged(box_num, 0);
    
    for (int i = 0; i < box_num; i++)
    {
        if (merged[i])
            continue;

        output.push_back(input[i]);

        float h0 = input[i].y2 - input[i].y1 + 1;
        float w0 = input[i].x2 - input[i].x1 + 1;

        float area0 = h0 * w0;


        for (int j = i + 1; j < box_num; j++)
        {
            if (merged[j])
                continue;

            float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;
            float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

            float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;
            float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

            float inner_h = inner_y1 - inner_y0 + 1;
            float inner_w = inner_x1 - inner_x0 + 1;


            if (inner_h <= 0 || inner_w <= 0)
                continue;

            float inner_area = inner_h * inner_w;

            float h1 = input[j].y2 - input[j].y1 + 1;
            float w1 = input[j].x2 - input[j].x1 + 1;

            float area1 = h1 * w1;

            float score;

            score = inner_area / (area0 + area1 - inner_area);

            if (score > iou_threshold)
                merged[j] = 1;
        }
    }
}
