#ifdef USE_DEPTH_DL

#include "depth/onnx_depth_estimator.h"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace svslam {

OnnxDepthEstimator::OnnxDepthEstimator(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "DepthEstimator"),
      session_(nullptr) {

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = Ort::Session(env_, model_path.c_str(), session_options);

    // Get input names
    size_t num_inputs = session_.GetInputCount();
    for (size_t i = 0; i < num_inputs; ++i) {
        auto name = session_.GetInputNameAllocated(i, allocator_);
        input_names_.push_back(name.get());
        input_name_ptrs_.push_back(std::move(name));
    }

    // Get output names
    size_t num_outputs = session_.GetOutputCount();
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name = session_.GetOutputNameAllocated(i, allocator_);
        output_names_.push_back(name.get());
        output_name_ptrs_.push_back(std::move(name));
    }

    std::cout << "OnnxDepthEstimator: Loaded model from " << model_path
              << " (inputs=" << num_inputs << ", outputs=" << num_outputs << ")" << std::endl;
}

void OnnxDepthEstimator::preprocess(const cv::Mat& image, std::vector<float>& blob) {
    // Convert to BGR if grayscale
    cv::Mat bgr;
    if (image.channels() == 1) {
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    } else if (image.channels() == 3) {
        bgr = image;
    } else {
        cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
    }

    // Resize to model input size
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(kInputW, kInputH), 0, 0, cv::INTER_LINEAR);

    // Convert to float [0, 1]
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // ImageNet normalization
    const float mean[] = {0.485f, 0.456f, 0.406f};  // BGR -> B, G, R
    const float stddev[] = {0.229f, 0.224f, 0.225f};

    // Split channels and normalize
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);

    // Note: OpenCV BGR order, Depth Anything expects RGB
    // channels[0]=B, channels[1]=G, channels[2]=R
    // Model input order: R, G, B (CHW)
    blob.resize(3 * kInputH * kInputW);

    for (int c = 0; c < 3; ++c) {
        // Map: c=0 -> R (channels[2]), c=1 -> G (channels[1]), c=2 -> B (channels[0])
        int cv_channel = 2 - c;
        const float m = mean[cv_channel];
        const float s = stddev[cv_channel];

        for (int y = 0; y < kInputH; ++y) {
            const float* row = channels[cv_channel].ptr<float>(y);
            for (int x = 0; x < kInputW; ++x) {
                blob[c * kInputH * kInputW + y * kInputW + x] = (row[x] - m) / s;
            }
        }
    }
}

cv::Mat OnnxDepthEstimator::postprocess(const std::vector<float>& output, int orig_h, int orig_w) {
    // Output is inverse depth (disparity) at model resolution
    cv::Mat disparity(kInputH, kInputW, CV_32FC1);
    std::memcpy(disparity.data, output.data(), output.size() * sizeof(float));

    // Convert disparity to depth
    cv::Mat depth(kInputH, kInputW, CV_32FC1);
    for (int y = 0; y < kInputH; ++y) {
        const float* disp_row = disparity.ptr<float>(y);
        float* depth_row = depth.ptr<float>(y);
        for (int x = 0; x < kInputW; ++x) {
            float d = disp_row[x];
            if (d > 1e-6f) {
                depth_row[x] = 1.0f / d;
            } else {
                depth_row[x] = 0.0f;
            }
        }
    }

    // Scale so median depth is ~1.5m (reasonable indoor assumption)
    std::vector<float> valid_depths;
    valid_depths.reserve(kInputH * kInputW);
    for (int y = 0; y < kInputH; ++y) {
        const float* row = depth.ptr<float>(y);
        for (int x = 0; x < kInputW; ++x) {
            if (row[x] > 0.0f && std::isfinite(row[x])) {
                valid_depths.push_back(row[x]);
            }
        }
    }

    if (!valid_depths.empty()) {
        std::sort(valid_depths.begin(), valid_depths.end());
        float median = valid_depths[valid_depths.size() / 2];
        if (median > 1e-6f) {
            float scale = 1.5f / median;
            depth *= scale;
        }
    }

    // Clamp to reasonable range
    for (int y = 0; y < kInputH; ++y) {
        float* row = depth.ptr<float>(y);
        for (int x = 0; x < kInputW; ++x) {
            if (row[x] < 0.1f || !std::isfinite(row[x])) {
                row[x] = 0.0f;
            } else if (row[x] > 20.0f) {
                row[x] = 0.0f;
            }
        }
    }

    // Resize to original resolution
    cv::Mat depth_orig;
    cv::resize(depth, depth_orig, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_LINEAR);

    return depth_orig;
}

cv::Mat OnnxDepthEstimator::estimate(const cv::Mat& image) {
    // Preprocess
    std::vector<float> blob;
    preprocess(image, blob);

    // Create input tensor
    std::array<int64_t, 4> input_shape = {1, 3, kInputH, kInputW};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, blob.data(), blob.size(), input_shape.data(), input_shape.size());

    // Run inference
    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names_.data(), &input_tensor, 1,
        output_names_.data(), output_names_.size());

    // Extract output
    const float* output_data = output_tensors[0].GetTensorData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    // Determine output size
    size_t output_size = 1;
    for (auto s : output_shape) output_size *= s;

    std::vector<float> output(output_data, output_data + output_size);

    return postprocess(output, image.rows, image.cols);
}

}

#endif // USE_DEPTH_DL
