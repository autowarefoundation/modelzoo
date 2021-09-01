// Copyright 2021 Apex.AI, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

/// \copyright Copyright 2021 Apex.AI, Inc.
/// All rights reserved.

#include <assert.h>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

// model files
#define network_dir "./"
#define network_module_path network_dir "deploy_lib.so"
#define network_graph_path network_dir "deploy_graph.json"
#define network_params_path network_dir "deploy_param.params"
// Name of file containing the human readable names of the classes. One class
// on each line.
#define LABEL_FILENAME "labels.txt"
// Name of file containing the anchor values for the network. Each line is one
// anchor. each anchor has 2 comma separated floating point values.
#define ANCHOR_FILENAME "anchors.csv"

// network configurations
#define TVDTYPE_CODE kDLFloat
#define TVDTYPE_BITS 32
#define TVDTYPE_LANES 1
#define TVDEVICE_TYPE kDLCPU
#define TVDEVICE_ID 0

// network input dimensions
#define NETWORK_INPUT_WIDTH 416
#define NETWORK_INPUT_HEIGHT 416
#define NETWORK_INPUT_DEPTH 3
#define NETWORK_INPUT_NAME "input"

// network output dimensions
#define NETWORK_OUTPUT_HEIGHT_1 13
#define NETWORK_OUTPUT_WIDTH_1 13
#define NETWORK_OUTPUT_DEPTH 255

// network output dimensions
#define NETWORK_OUTPUT_HEIGHT_2 26
#define NETWORK_OUTPUT_WIDTH_2 26

// network output dimensions
#define NETWORK_OUTPUT_HEIGHT_3 52
#define NETWORK_OUTPUT_WIDTH_3 52

// minimum confidence score by which to filter the output detections
#define SCORE_THRESHOLD 0.5

// threshold for filtering out candidate bounding boxes when IoU overlap exceeds this value
#define NMS_THRESHOLD 0.45

// filename of the image on which to run the inference
#define IMAGE_FILENAME "test_image_0.jpg"

// Name of the window to display the detection results
#define DISPLAY_WINDOW_NAME "YOLO Output"

/// Struct for storing detections in image with confidence scores
struct BoundingBox {
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float conf;

    BoundingBox(float x_min, float y_min, float x_max, float y_max, float conf_) : xmin(x_min), ymin(y_min), xmax(x_max), ymax(y_max), conf(conf_){};
};

/// Map for storing the class and detections in the image
std::map<int, std::vector<BoundingBox>> bbox_map{};
static const int NETWORK_OUTPUT_WIDTH[3] = {NETWORK_OUTPUT_WIDTH_1, NETWORK_OUTPUT_WIDTH_2, NETWORK_OUTPUT_WIDTH_3};
static const int NETWORK_OUTPUT_HEIGHT[3] = {NETWORK_OUTPUT_HEIGHT_1, NETWORK_OUTPUT_HEIGHT_2, NETWORK_OUTPUT_HEIGHT_3};

/// \brief gets the amount of overlap between the two bounding boxes
/// \return returns the overlap amount
float interval_overlap(std::pair<float, float> interval_a, std::pair<float, float> interval_b) {
    float x1 = interval_a.first;
    float x2 = interval_a.second;
    float x3 = interval_b.first;
    float x4 = interval_b.second;

    if (x3 < x1) {
        if (x4 < x1)
            return 0.0;
        else
            return std::min(x2, x4) - x1;
    }

    else {
        if (x2 < x3)
            return 0.0;
        else
            return std::min(x2, x4) - x3;
    }
}

/// \brief gets the IoU between the two bounding boxes
/// \return returns the IoU
float bbox_iou(const BoundingBox &box1, const BoundingBox &box2) {
    float intersect_w = interval_overlap(std::make_pair(box1.xmin, box1.xmax), std::make_pair(box2.xmin, box2.xmax));
    float intersect_h = interval_overlap(std::make_pair(box1.ymin, box1.ymax), std::make_pair(box2.ymin, box2.ymax));

    float intersect = intersect_w * intersect_h;

    float w1 = box1.xmax - box1.xmin;
    float h1 = box1.ymax - box1.ymin;
    float w2 = box2.xmax - box2.xmin;
    float h2 = box2.ymax - box2.ymin;

    float union_bbox = w1 * h1 + w2 * h2 - intersect;
    if (union_bbox == 0) {
        return 1.0;
    }
    return intersect / union_bbox;
}

/// \brief removes the duplicate detections with lower confidence scores for the same object
void do_nms() {
    for (auto &entry : bbox_map) {
        sort(entry.second.begin(), entry.second.end(), [](const BoundingBox &left_bbox, const BoundingBox &right_bbox) { return (left_bbox.conf > right_bbox.conf); });
    }

    for (auto &entry : bbox_map) {
        auto &bboxes = entry.second;
        for (int i = 0; i < bboxes.size(); i++) {
            if (bboxes[i].conf == 0)
                continue;

            for (int j = i + 1; j < bboxes.size(); j++) {
                if (bbox_iou(bboxes[i], bboxes[j]) >= NMS_THRESHOLD) {
                    bboxes[j].conf = 0;
                }
            }
        }
    }
}

int main(int argc, char const *argv[]) {
    // load compiled functions
    tvm::runtime::Module mod =
        tvm::runtime::Module::LoadFromFile(network_module_path);

    // load json graph
    std::ifstream json_in(network_graph_path, std::ios::in);
    std::string json_data((std::istreambuf_iterator<char>(json_in)),
                          std::istreambuf_iterator<char>());
    json_in.close();

    // load parameters from binary file
    std::ifstream params_in(network_params_path, std::ios::binary);
    std::string params_data((std::istreambuf_iterator<char>(params_in)),
                            std::istreambuf_iterator<char>());
    params_in.close();
    // parameters need to be in TVMByteArray format
    TVMByteArray params_arr;
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();

    // Create tvm runtime module
    tvm::runtime::Module runtime_mod =
        (*tvm::runtime::Registry::Get("tvm.graph_executor.create"))(json_data, mod, (int)TVDEVICE_TYPE, TVDEVICE_ID);

    // load parameters
    auto load_params = runtime_mod.GetFunction("load_params");
    load_params(params_arr);

    // get set_input function
    auto set_input = runtime_mod.GetFunction("set_input");

    // get the function which executes the network
    auto execute = runtime_mod.GetFunction("run");

    // get the function to get output data
    auto get_output = runtime_mod.GetFunction("get_output");

    // allocate input variable
    DLTensor *x{};
    int64_t shape_x[] = {1, NETWORK_INPUT_WIDTH, NETWORK_INPUT_HEIGHT,
                         NETWORK_INPUT_DEPTH};
    TVMArrayAlloc(shape_x, sizeof(shape_x) / sizeof(shape_x[0]), TVDTYPE_CODE,
                  TVDTYPE_BITS, TVDTYPE_LANES, TVDEVICE_TYPE, TVDEVICE_ID,
                  &x);

    // read input image
    auto image = cv::imread(IMAGE_FILENAME, cv::IMREAD_COLOR);
    // Compute the ratio for resizing and size for padding
    double scale_x =
        static_cast<double>(image.size().width) / NETWORK_INPUT_WIDTH;
    double scale_y =
        static_cast<double>(image.size().height) / NETWORK_INPUT_HEIGHT;
    double scale = std::max(scale_x, scale_y);

    // perform padding
    if (scale != 1) {
        cv::resize(image, image, cv::Size(), 1.0f / scale, 1.0f / scale);
    }

    size_t w_pad = NETWORK_INPUT_WIDTH - image.size().width;
    size_t h_pad = NETWORK_INPUT_HEIGHT - image.size().height;

    if (w_pad || h_pad) {
        cv::copyMakeBorder(image, image, h_pad / 2, (h_pad - h_pad / 2), w_pad / 2,
                           (w_pad - w_pad / 2), cv::BORDER_CONSTANT,
                           cv::Scalar(0, 0, 0));
    }

    // convert pixel values from int8 to float32. convert pixel value range from
    // 0 - 255 to 0 - 1.
    cv::Mat3f image_3f{};

    image.convertTo(image_3f, CV_32FC3, 1 / 255.0f);

    // cv library use BGR as a default color format, the network expects the data
    // in RGB format
    cv::cvtColor(image_3f, image_3f, cv::COLOR_BGR2RGB);
    TVMArrayCopyFromBytes(x, image_3f.data,
                          NETWORK_INPUT_HEIGHT * NETWORK_INPUT_WIDTH *
                              NETWORK_INPUT_DEPTH * sizeof(TVDTYPE_CODE));

    // parse human readable names for the classes
    std::ifstream label_file{LABEL_FILENAME};
    if (not label_file.good()) {
        std::cout << "unable to open label file:" << LABEL_FILENAME << std::endl;
    }
    std::vector<std::string> labels{};
    std::string line{};
    while (std::getline(label_file, line)) {
        labels.push_back(line);
    }
    // Get anchor values for this network from the anchor file
    std::ifstream anchor_file{ANCHOR_FILENAME};
    if (not anchor_file.good()) {
        std::cout << "unable to open anchor file:" << ANCHOR_FILENAME << std::endl;
    }
    std::string first{};
    std::string second{};
    std::vector<std::pair<float, float>> anchors{};
    while (std::getline(anchor_file, line)) {
        std::stringstream line_stream(line);
        std::getline(line_stream, first, ',');
        std::getline(line_stream, second, ',');
        anchors.push_back(
            std::make_pair(std::atof(first.c_str()), std::atof(second.c_str())));
    }

    // execute the inference
    set_input(NETWORK_INPUT_NAME, x);
    execute();

    // Loop through for the multiscale detections
    for (int scale = 0; scale < 3; scale++) {
        // allocate output variable
        DLTensor *y{};
        int64_t shape_y[] = {1, NETWORK_OUTPUT_WIDTH[scale], NETWORK_OUTPUT_HEIGHT[scale], NETWORK_OUTPUT_DEPTH};
        TVMArrayAlloc(shape_y, sizeof(shape_y) / sizeof(shape_y[0]), TVDTYPE_CODE,
                      TVDTYPE_BITS, TVDTYPE_LANES, kDLCPU, 0,
                      &y);
        get_output(scale, y);
        auto l_h = shape_y[1];           // layer height
        auto l_w = shape_y[2];           // layer width
        auto n_classes = labels.size();  // total number of classes
        auto n_anchors = 3;              // total number of anchors
        auto n_coords = 4;               // number of coordinates in a single anchor box
        auto nudetections = n_classes * n_anchors * l_w * l_h;

        // assert data is stored row-majored in y and the dtype is float
        assert(y->strides == nullptr);
        assert(y->dtype.bits == sizeof(float) * 8);

        // get a pointer to the output data
        float *data_ptr = (float *)((uint8_t *)y->data + y->byte_offset);

        // utility function to return data from y given index
        auto get_output_data = [data_ptr, shape_y, n_classes, n_anchors,
                                n_coords](auto row_i, auto col_j, auto anchor_k,
                                          auto offset) {
            auto box_index = (row_i * shape_y[2] + col_j) * shape_y[3];
            auto index = box_index + anchor_k * (n_classes + n_coords + 1);
            return data_ptr[index + offset];
        };

        // sigmoid function
        auto sigmoid = [](float x) { return (float)(1.0 / (1.0 + std::exp(-x))); };

        // Parse results into detections. Loop over each detection cell in the model
        // output
        for (size_t i = 0; i < l_w; i++) {
            for (size_t j = 0; j < l_h; j++) {
                for (size_t anchor_k = scale * n_anchors; anchor_k < (scale + 1) * n_anchors; anchor_k++) {
                    float anchor_w = anchors[anchor_k].first;
                    float anchor_h = anchors[anchor_k].second;

                    // Compute property indices
                    auto box_x = get_output_data(i, j, anchor_k, 0);
                    auto box_y = get_output_data(i, j, anchor_k, 1);
                    auto box_w = get_output_data(i, j, anchor_k, 2);
                    auto box_h = get_output_data(i, j, anchor_k, 3);
                    auto box_p = get_output_data(i, j, anchor_k, 4);

                    // Transform log-space predicted coordinates to absolute space + offset
                    // Transform bounding box position from offset to absolute (ratio)
                    auto x_coord = (sigmoid(box_x) + j) / l_w;
                    auto y_coord = (sigmoid(box_y) + i) / l_h;

                    // Transform bounding box height and width from log to absolute space
                    auto w = anchor_w * exp(box_w) / NETWORK_INPUT_WIDTH;
                    auto h = anchor_h * exp(box_h) / NETWORK_INPUT_HEIGHT;

                    // Decode the confidence of detection in this anchor box
                    auto p_0 = sigmoid(box_p);

                    // find maximum probability of all classes
                    float max_p = 0.0f;
                    int max_ind = -1;
                    for (int i_class = 0; i_class < n_classes; i_class++) {
                        auto class_p = get_output_data(i, j, anchor_k, 5 + i_class);
                        if (max_p < class_p) {
                            max_p = class_p;
                            max_ind = i_class;
                        }
                    }

                    // decode and copy class probabilities
                    std::vector<float> class_probabilities{};
                    float p_total = 0;
                    for (size_t i_class = 0; i_class < n_classes; i_class++) {
                        auto class_p = get_output_data(i, j, anchor_k, 5 + i_class);
                        class_probabilities.push_back(std::exp(class_p - max_p));
                        p_total += class_probabilities[i_class];
                    }

                    // Find the most likely score
                    auto max_score = class_probabilities[max_ind] * p_0 / p_total;

                    if (max_score > SCORE_THRESHOLD) {
                        if (bbox_map.count(max_ind) == 0) {
                            bbox_map[max_ind] = std::vector<BoundingBox>{};
                        }
                        bbox_map[max_ind].push_back(BoundingBox(x_coord - w / 2, y_coord - h / 2, x_coord + w / 2, y_coord + h / 2, max_score));
                    }
                }
            }
        }
    }

    // suppress non-maximal boxes
    do_nms();
    // draw all detections with high scores
    for (const auto &entry : bbox_map) {
        for (const auto &bbox : entry.second) {
            if (bbox.conf == 0)
                continue;
            float original_w = (bbox.xmax - bbox.xmin) * NETWORK_INPUT_WIDTH;
            float original_h = (bbox.ymax - bbox.ymin) * NETWORK_INPUT_HEIGHT;
            float original_x = bbox.xmin * NETWORK_INPUT_WIDTH;
            float original_y = bbox.ymin * NETWORK_INPUT_HEIGHT;

            // draw rectangle on image
            cv::rectangle(
                image, cv::Rect(original_x, original_y, original_w, original_h),
                cv::Scalar(0, 255, 0));

            // draw the class text
            cv::putText(image, labels[entry.first],
                        cv::Point(original_x + 1, original_y + 15),
                        cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0), 1.5);

            // draw the score text
            cv::putText(image, std::to_string(bbox.conf).substr(0, 5),
                        cv::Point(original_x + original_w - 48, original_y + 15),
                        cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0), 1.5);
        }
    }
    // show in a pop up window the detection results
    cv::namedWindow(DISPLAY_WINDOW_NAME, cv::WINDOW_AUTOSIZE);
    cv::imshow(DISPLAY_WINDOW_NAME, image);

    // wait for user to close the window
    cv::waitKey(0);

    return 0;
}