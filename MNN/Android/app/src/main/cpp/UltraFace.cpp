//  Created by Linzaer on 2019/11/15.
//  Copyright © 2019 Linzaer. All rights reserved.

#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

#include "UltraFace.hpp"
#define TAG "cpp"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
using namespace std;

UltraFace::UltraFace(std::string &mnn_path,
                     int input_width, int input_length, int num_thread_,
                     float score_threshold_, float iou_threshold_) {
    num_thread = num_thread_;
    score_threshold = score_threshold_;
    iou_threshold = iou_threshold_;
    in_w = input_width;
    in_h = input_length;
    w_h_list = {in_w, in_h};

    for (auto size : w_h_list) {
        std::vector<float> fm_item;
        for (float stride : strides) {
            fm_item.push_back(ceil(size / stride));
        }
        featuremap_size.push_back(fm_item);
    }

    for (auto size : w_h_list) {
        shrinkage_size.push_back(strides);
    }
    /* generate prior anchors */
    for (int index = 0; index < num_featuremap; index++) {
        float scale_w = in_w / shrinkage_size[0][index];
        float scale_h = in_h / shrinkage_size[1][index];
        for (int j = 0; j < featuremap_size[1][index]; j++) {
            for (int i = 0; i < featuremap_size[0][index]; i++) {
                float x_center = (i + 0.5) / scale_w;
                float y_center = (j + 0.5) / scale_h;

                for (float k : min_boxes[index]) {
                    float w = k / in_w;
                    float h = k / in_h;
                    priors.push_back({clip(x_center, 1), clip(y_center, 1), clip(w, 1), clip(h, 1)});
                }
            }
        }
    }
    /* generate prior anchors finished */

    num_anchors = priors.size();

    ultra_net.load_param(mnn_path, num_thread);
    ultra_net.set_params(1, 1, mean_vals, norm_vals);


}

int UltraFace::detect(cv::Mat &raw_image, std::vector<FaceInfo> &face_list ) {
    if (raw_image.empty()) {
        LOGD("raw_image image is null");
        return -1;
    }

    image_h = raw_image.rows;
    image_w = raw_image.cols;
    cv::Mat image;
    LOGD("----------------");
    //cv::flip(raw_image, raw_image, 1);
    cv::resize(raw_image, image, cv::Size(in_w, in_h));

    Inference_engine_tensor  out;


    string scores = "scores";
    out.add_name(scores);


    string boxes = "boxes";
    out.add_name(boxes);


    ultra_net.infer_img(image, out);
    // get output data

//    for(int i=0 ; i < 1000; i++)
//    {
//        LOGD("score11 = %f", out.score(0)[i  ]);
//        if (out.score(0)[i * 2 + 1 ] > score_threshold) {
//            LOGD("score = %f", out.score(0)[i * 2 + 1 ]);
//        }
//    }

    std::vector<FaceInfo> bbox_collection;
    generateBBox(bbox_collection, out.score(0) , out.score(1));
    //LOGD("bbox_collection == %d", bbox_collection.size());
    nms(bbox_collection, face_list);
    return 0;
}
//void UltraFace::generateBBox(std::vector<FaceInfo> &bbox_collection, float* scores, float* boxes) {
//    for (int i = 0; i < num_anchors; i++) {
//        if (scores[i * 2 + 1 ] > score_threshold) {
//            //LOGD("score = %f",scores[i * 2 + 1 ]);
//
//            FaceInfo rects;
//            float x_center = boxes[i * 4] * center_variance * priors[i][2] + priors[i][0];
//            float y_center = boxes[i * 4 + 1] * center_variance * priors[i][3] + priors[i][1];
//            float w = exp(boxes[i * 4 + 2] * size_variance) * priors[i][2];
//            float h = exp(boxes[i * 4 + 3] * size_variance) * priors[i][3];
//
//            rects.x1 = clip(x_center - w / 2.0, 1) * image_w;
//            rects.y1 = clip(y_center - h / 2.0, 1) * image_h;
//            rects.x2 = clip(x_center + w / 2.0, 1) * image_w;
//            rects.y2 = clip(y_center + h / 2.0, 1) * image_h;
//            rects.score = clip(scores[i * 2 + 1 ], 1);
//
//            bbox_collection.push_back(rects);
//        }
//    }
//}

void UltraFace::generateBBox(std::vector<FaceInfo> &bbox_collection, std::vector<float> scores, std::vector<float> boxes) {
    for (int i = 0; i < num_anchors; i++) {
        if (scores[i * 2 + 1 ] > score_threshold) {
            //LOGD("score = %f",scores[i * 2 + 1 ]);

            FaceInfo rects;
            float x_center = boxes[i * 4] * center_variance * priors[i][2] + priors[i][0];
            float y_center = boxes[i * 4 + 1] * center_variance * priors[i][3] + priors[i][1];
            float w = exp(boxes[i * 4 + 2] * size_variance) * priors[i][2];
            float h = exp(boxes[i * 4 + 3] * size_variance) * priors[i][3];

            rects.x1 = clip(x_center - w / 2.0, 1) * image_w;
            rects.y1 = clip(y_center - h / 2.0, 1) * image_h;
            rects.x2 = clip(x_center + w / 2.0, 1) * image_w;
            rects.y2 = clip(y_center + h / 2.0, 1) * image_h;
            rects.score = clip(scores[i * 2 + 1 ], 1);

            bbox_collection.push_back(rects);
        }
    }
}

void UltraFace::nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, int type) {
    std::sort(input.begin(), input.end(), [](const FaceInfo &a, const FaceInfo &b) { return a.score > b.score; });

    int box_num = input.size();
    //LOGD("box_num == %d",box_num);
    std::vector<int> merged(box_num, 0);

    for (int i = 0; i < box_num; i++) {
        if (merged[i])
            continue;
        std::vector<FaceInfo> buf;

        buf.push_back(input[i]);
        merged[i] = 1;

        float h0 = input[i].y2 - input[i].y1 + 1;
        float w0 = input[i].x2 - input[i].x1 + 1;

        float area0 = h0 * w0;

        for (int j = i + 1; j < box_num; j++) {
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

            if (score > iou_threshold) {
                merged[j] = 1;
                buf.push_back(input[j]);
            }
        }
        switch (type) {
            case hard_nms: {
                output.push_back(buf[0]);
                break;
            }
            case blending_nms: {
                float total = 0;
                for (int i = 0; i < buf.size(); i++) {
                    total += exp(buf[i].score);
                }
                FaceInfo rects;
                memset(&rects, 0, sizeof(rects));
                for (int i = 0; i < buf.size(); i++) {
                    float rate = exp(buf[i].score) / total;
                    rects.x1 += buf[i].x1 * rate;
                    rects.y1 += buf[i].y1 * rate;
                    rects.x2 += buf[i].x2 * rate;
                    rects.y2 += buf[i].y2 * rate;
                    rects.score += buf[i].score * rate;
                }
                output.push_back(rects);
                break;
            }
            default: {
                printf("wrong type of nms.");
                exit(-1);
            }
        }
    }
}
