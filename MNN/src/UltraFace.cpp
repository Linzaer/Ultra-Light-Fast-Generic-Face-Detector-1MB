//  Created by Linzaer on 2019/11/15.
//  Copyright © 2019 Linzaer. All rights reserved.

#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

#include "UltraFace.hpp"

using namespace std;

UltraFace::UltraFace(const std::string &mnn_path,
                     int input_size, int num_thread_,
                     float score_threshold_, float iou_threshold_, int topk_) {
    num_thread = num_thread_;
    score_threshold = score_threshold_;
    iou_threshold = iou_threshold_;

    switch (input_size) {
        case 128: {
            in_w = 128;
            in_h = 96;
            num_anchors = 708;
            featuremap_size = {{16, 8, 4, 2},
                               {12, 6, 3, 2}};
            break;
        }
        case 160: {
            in_w = 160;
            in_h = 120;
            num_anchors = 1118;
            featuremap_size = {{20, 10, 5, 3},
                               {15, 8,  4, 2}};
            break;
        }
        case 320: {
            in_w = 320;
            in_h = 240;
            num_anchors = 4420;
            featuremap_size = {{40, 20, 10, 5},
                               {30, 15, 8,  4}};
            break;
        }
        case 480: {
            in_w = 480;
            in_h = 360;
            num_anchors = 9984;
            featuremap_size = {{60, 30, 15, 8},
                               {45, 23, 12, 6}};
            break;
        }
        case 640: {
            in_w = 640;
            in_h = 480;
            num_anchors = 17640;
            featuremap_size = {{80, 40, 20, 10},
                               {60, 30, 15, 8}};
            break;
        }
        case 1280: {
            in_w = 1280;
            in_h = 960;
            num_anchors = 70500;
            featuremap_size = {{160, 80, 40, 20},
                               {120, 60, 30, 15}};
            break;
        }
        default: {
            printf("unknown input size.");
            exit(-1);
        }
    }
    w_h_list = {in_w, in_h};
    for (int i = 0; i < 2; ++i) {
        std::vector<float> shrinkage_item;
        for (int j = 0; j < featuremap_size[i].size(); ++j) {
            shrinkage_item.push_back(w_h_list[i] / featuremap_size[i][j]);
        }
        shrinkage_size.push_back(shrinkage_item);
    }

    /* generate prior anchors */
    for (int index = 0; index < num_featuremap; index++) {
        float scale_w = in_w / shrinkage_size[0][index];
        float scale_h = in_h / shrinkage_size[1][index];
        for (int j = 0; j < featuremap_size[1][index]; j++) {
            for (int i = 0; i < featuremap_size[0][index]; i++) {
                float x_center = (i + 0.5) / scale_w;
                float y_center = (j + 0.5) / scale_h;

                for (int k = 0; k < min_boxes[index].size(); k++) {
                    float w = min_boxes[index][k] / in_w;
                    float h = min_boxes[index][k] / in_h;
                    priors.push_back({clip(x_center, 1), clip(y_center, 1), clip(w, 1), clip(h, 1)});
                }
            }
        }
    }
    /* generate prior anchors finished */



    ultraface_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path.c_str()));
    MNN::ScheduleConfig config;
    config.numThread = num_thread;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode) 2;
    config.backendConfig = &backendConfig;

    ultraface_session = ultraface_interpreter->createSession(config);

    input_tensor = ultraface_interpreter->getSessionInput(ultraface_session, nullptr);
    auto shape = input_tensor->shape();

}

UltraFace::~UltraFace() {
    ultraface_interpreter->releaseModel();
    ultraface_interpreter->releaseSession(ultraface_session);
}

int UltraFace::detect(cv::Mat &raw_image, std::vector<FaceInfo> &face_list) {
    if (raw_image.empty()) {
        std::cout << "image is empty ,please check!" << std::endl;
        return -1;
    }

    image_h = raw_image.rows;
    image_w = raw_image.cols;
    cv::Mat image;
    cv::resize(raw_image, image, cv::Size(in_w, in_h));


    std::shared_ptr<MNN::CV::ImageProcess> pretreat(
            MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, mean_vals, 3,
                                          norm_vals, 3));
    pretreat->convert(image.data, in_w, in_h, image.step[0], input_tensor);

    auto start = chrono::steady_clock::now();


    // run network
    ultraface_interpreter->runSession(ultraface_session);

    // get output data

    string scores = "scores";
    string boxes = "boxes";
    MNN::Tensor *tensor_scores = ultraface_interpreter->getSessionOutput(ultraface_session, scores.c_str());
    MNN::Tensor *tensor_boxes = ultraface_interpreter->getSessionOutput(ultraface_session, boxes.c_str());

    MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());

    tensor_scores->copyToHostTensor(&tensor_scores_host);

    MNN::Tensor tensor_boxes_host(tensor_boxes, tensor_boxes->getDimensionType());

    tensor_boxes->copyToHostTensor(&tensor_boxes_host);

    std::vector<FaceInfo> bbox_collection;


    auto end = chrono::steady_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "inference time:" << elapsed.count() << " s" << endl;

    generateBBox(bbox_collection, tensor_scores, tensor_boxes);
    nms(bbox_collection, face_list);
    return 0;
}

void UltraFace::generateBBox(std::vector<FaceInfo> &bbox_collection, MNN::Tensor *scores, MNN::Tensor *boxes) {
    for (int i = 0; i < num_anchors; i++) {
        if (scores->host<float>()[i * 2 + 1] > score_threshold) {
            FaceInfo rects;
            float x_center = boxes->host<float>()[i * 4] * center_variance * priors[i][2] + priors[i][0];
            float y_center = boxes->host<float>()[i * 4 + 1] * center_variance * priors[i][3] + priors[i][1];
            float w = exp(boxes->host<float>()[i * 4 + 2] * size_variance) * priors[i][2];
            float h = exp(boxes->host<float>()[i * 4 + 3] * size_variance) * priors[i][3];

            rects.x1 = clip(x_center - w / 2.0, 1) * image_w;
            rects.y1 = clip(y_center - h / 2.0, 1) * image_h;
            rects.x2 = clip(x_center + w / 2.0, 1) * image_w;
            rects.y2 = clip(y_center + h / 2.0, 1) * image_h;
            rects.score = clip(scores->host<float>()[i * 2 + 1], 1);
            bbox_collection.push_back(rects);
        }
    }
}

void UltraFace::nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, int type) {
    std::sort(input.begin(), input.end(), [](const FaceInfo &a, const FaceInfo &b) { return a.score > b.score; });

    int box_num = input.size();

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
