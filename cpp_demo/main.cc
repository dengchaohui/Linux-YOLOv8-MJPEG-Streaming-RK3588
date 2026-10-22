// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <opencv2/opencv.hpp>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "yolov8.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("%s <model_path> <device_path>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *device_path = argv[2];   // e.g. /dev/video11

    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();

    ret = init_yolov8_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
        return -1;
    }

    // ===== GStreamer 打开摄像头 =====
    cv::VideoCapture cap;
    std::string pipeline =
        "v4l2src device=" + std::string(device_path) +
        " ! video/x-raw, format=NV12, width=1280, height=720, framerate=30/1"
        " ! videoconvert"
        " ! video/x-raw, format=BGR"
        " ! appsink";
    printf("GStreamer pipeline: %s\n", pipeline.c_str());
    cap.open(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        printf("Failed to open camera: %s\n", device_path);
        return -1;
    }

    // ===== 抓 10 帧，逐帧推理、画框、存图 =====
    int frame_cnt = 0;
    cv::Mat src_frame;
    while (frame_cnt < 10) {
        cap >> src_frame;
        if (src_frame.empty()) {
            printf("Failed to grab frame\n");
            break;
        }

        // 构造 image_buffer_t 交给推理
        image_buffer_t src_image;
        memset(&src_image, 0, sizeof(image_buffer_t));
        src_image.height = src_frame.rows;
        src_image.width  = src_frame.cols;
        src_image.width_stride = src_frame.step[0];
        src_image.virt_addr = src_frame.data;
        src_image.format = IMAGE_FORMAT_RGB888;
        src_image.size = src_frame.total() * src_frame.elemSize();

        object_detect_result_list od_results;
        ret = inference_yolov8_model(&rknn_app_ctx, &src_image, &od_results);
        if (ret != 0) {
            printf("inference_yolov8_model fail! ret=%d\n", ret);
            break;
        }

        // 画框
        char text[256];
        for (int i = 0; i < od_results.count; i++) {
            object_detect_result *det_result = &(od_results.results[i]);
            printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
                   det_result->box.left, det_result->box.top,
                   det_result->box.right, det_result->box.bottom, det_result->prop);

            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;

            cv::rectangle(src_frame, cv::Point(x1, y1), cv::Point(x2, y2),
                          cv::Scalar(255, 0, 0), 3);
            sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id),
                    det_result->prop * 100);
            cv::putText(src_frame, text, cv::Point(x1, y1 - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        cv::Scalar(0, 0, 255), 2);
        }

        // 存图
        char fname[64];
        sprintf(fname, "/userdata/rknn_yolov8_demo_new/out_cam_%02d.jpg", frame_cnt);
        cv::imwrite(fname, src_frame);
        printf("saved %s, det_count=%d\n", fname, od_results.count);

        frame_cnt++;
    }

    cap.release();

    deinit_post_process();

    ret = release_yolov8_model(&rknn_app_ctx);
    if (ret != 0) {
        printf("release_yolov8_model fail! ret=%d\n", ret);
    }

    return 0;
}
