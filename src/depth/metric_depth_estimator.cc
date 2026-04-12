#ifdef USE_DEPTH_DL

#include "depth/metric_depth_estimator.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

int64_t resolveDynamicDim(int64_t dim, int64_t fallback) {
    return dim > 0 ? dim : fallback;
}

std::size_t checkedElementCount(const std::vector<int64_t>& shape) {
    std::size_t count = 1;
    for (const int64_t dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error("MetricDepthEstimator: unresolved dynamic tensor shape");
        }
        count *= static_cast<std::size_t>(dim);
    }
    return count;
}

bool looksLikeImageName(const std::string& lower_name) {
    return lower_name.find("image") != std::string::npos ||
           lower_name.find("rgb") != std::string::npos ||
           lower_name.find("pixel") != std::string::npos ||
           lower_name == "input";
}

bool looksLikeDepthName(const std::string& lower_name) {
    return lower_name.find("depth") != std::string::npos ||
           lower_name.find("pred") != std::string::npos;
}

bool looksLikeNonDepthName(const std::string& lower_name) {
    return lower_name.find("confidence") != std::string::npos ||
           lower_name.find("normal") != std::string::npos ||
           lower_name.find("mask") != std::string::npos;
}

bool looksLikeSizeName(const std::string& lower_name) {
    return lower_name.find("size") != std::string::npos ||
           lower_name.find("shape") != std::string::npos ||
           lower_name.find("resolution") != std::string::npos ||
           lower_name.find("hw") != std::string::npos;
}

bool looksLikeOriginalSizeName(const std::string& lower_name) {
    return lower_name.find("orig") != std::string::npos ||
           lower_name.find("source") != std::string::npos ||
           lower_name.find("raw") != std::string::npos;
}

bool looksLikeIntrinsicName(const std::string& lower_name) {
    return lower_name == "k" ||
           lower_name.find("intrinsic") != std::string::npos ||
           lower_name.find("camera") != std::string::npos ||
           lower_name.find("cam") != std::string::npos;
}

}  // namespace

namespace svslam {

MetricDepthEstimator::MetricDepthEstimator(const std::string& model_path, Camera::Ptr camera)
    : camera_(std::move(camera)),
      env_(ORT_LOGGING_LEVEL_WARNING, "MetricDepthEstimator"),
      session_(nullptr) {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = Ort::Session(env_, model_path.c_str(), session_options);

    const size_t num_inputs = session_.GetInputCount();
    for (size_t i = 0; i < num_inputs; ++i) {
        auto name = session_.GetInputNameAllocated(i, allocator_);
        input_names_.push_back(name.get());
        input_name_ptrs_.push_back(std::move(name));

        const auto tensor_info = session_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
        input_shapes_.push_back(tensor_info.GetShape());
        input_types_.push_back(tensor_info.GetElementType());
    }

    const size_t num_outputs = session_.GetOutputCount();
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name = session_.GetOutputNameAllocated(i, allocator_);
        output_names_.push_back(name.get());
        output_name_ptrs_.push_back(std::move(name));

        const auto tensor_info = session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
        output_shapes_.push_back(tensor_info.GetShape());
        output_types_.push_back(tensor_info.GetElementType());
    }

    image_input_index_ = findImageInputIndex();
    depth_output_index_ = findDepthOutputIndex();

    std::cout << "MetricDepthEstimator: Loaded model from " << model_path
              << " (inputs=" << num_inputs << ", outputs=" << num_outputs
              << ", image_input=" << input_names_[image_input_index_]
              << ", depth_output=" << output_names_[depth_output_index_] << ")" << std::endl;
}

MetricDepthEstimator::MetricDepthEstimator(Camera::Ptr camera)
    : camera_(std::move(camera)),
      env_(ORT_LOGGING_LEVEL_WARNING, "MetricDepthEstimator"),
      session_(nullptr) {}

cv::Size MetricDepthEstimator::resolveInputSize(const std::vector<int64_t>& input_shape,
                                                const cv::Size& image_size) {
    if (input_shape.size() != 3 && input_shape.size() != 4) {
        throw std::runtime_error("MetricDepthEstimator: unsupported image input tensor rank");
    }

    const bool nhwc = isNhwcImageShape(input_shape);
    int height = image_size.height;
    int width = image_size.width;

    if (input_shape.size() == 4) {
        if (nhwc) {
            height = static_cast<int>(resolveDynamicDim(input_shape[1], image_size.height));
            width = static_cast<int>(resolveDynamicDim(input_shape[2], image_size.width));
        } else {
            height = static_cast<int>(resolveDynamicDim(input_shape[2], image_size.height));
            width = static_cast<int>(resolveDynamicDim(input_shape[3], image_size.width));
        }
    } else {
        if (nhwc) {
            height = static_cast<int>(resolveDynamicDim(input_shape[0], image_size.height));
            width = static_cast<int>(resolveDynamicDim(input_shape[1], image_size.width));
        } else {
            height = static_cast<int>(resolveDynamicDim(input_shape[1], image_size.height));
            width = static_cast<int>(resolveDynamicDim(input_shape[2], image_size.width));
        }
    }

    if (height <= 0 || width <= 0) {
        throw std::runtime_error("MetricDepthEstimator: invalid image input tensor shape");
    }
    return cv::Size(width, height);
}

cv::Size MetricDepthEstimator::resolveOutputSize(const std::vector<int64_t>& output_shape) {
    std::vector<int64_t> spatial_dims;
    spatial_dims.reserve(output_shape.size());
    for (const int64_t dim : output_shape) {
        if (dim > 1) {
            spatial_dims.push_back(dim);
        }
    }

    if (spatial_dims.size() != 2) {
        throw std::runtime_error("MetricDepthEstimator: unsupported depth output tensor shape");
    }

    return cv::Size(static_cast<int>(spatial_dims[1]), static_cast<int>(spatial_dims[0]));
}

bool MetricDepthEstimator::isNhwcImageShape(const std::vector<int64_t>& shape) {
    if (shape.size() == 4) {
        if (shape[3] == 3) {
            return true;
        }
        if (shape[1] == 3) {
            return false;
        }
    } else if (shape.size() == 3) {
        if (shape[2] == 3) {
            return true;
        }
        if (shape[0] == 3) {
            return false;
        }
    }
    return false;
}

size_t MetricDepthEstimator::findImageInputIndex() const {
    size_t fallback_index = input_names_.size();

    for (size_t i = 0; i < input_names_.size(); ++i) {
        const auto lower_name = toLower(input_names_[i]);
        const auto& shape = input_shapes_[i];

        if (input_types_[i] != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            continue;
        }
        if (shape.size() != 3 && shape.size() != 4) {
            continue;
        }

        if (looksLikeImageName(lower_name)) {
            return i;
        }

        const bool nchw = (shape.size() == 4 && shape[1] == 3) ||
                          (shape.size() == 3 && shape[0] == 3);
        const bool nhwc = (shape.size() == 4 && shape[3] == 3) ||
                          (shape.size() == 3 && shape[2] == 3);
        if (nchw || nhwc) {
            fallback_index = std::min(fallback_index, i);
        }
    }

    if (fallback_index != input_names_.size()) {
        return fallback_index;
    }

    throw std::runtime_error("MetricDepthEstimator: could not identify image input tensor");
}

size_t MetricDepthEstimator::findDepthOutputIndex() const {
    int best_score = std::numeric_limits<int>::min();
    size_t best_index = output_names_.size();

    for (size_t i = 0; i < output_names_.size(); ++i) {
        if (output_types_[i] != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            continue;
        }

        int score = 0;
        const auto lower_name = toLower(output_names_[i]);
        if (looksLikeDepthName(lower_name)) {
            score += 100;
        }
        if (looksLikeNonDepthName(lower_name)) {
            score -= 100;
        }

        try {
            const cv::Size output_size = resolveOutputSize(output_shapes_[i]);
            if (output_size.width > 0 && output_size.height > 0) {
                score += 10;
            }
        } catch (const std::exception&) {
            continue;
        }

        if (score > best_score) {
            best_score = score;
            best_index = i;
        }
    }

    if (best_index != output_names_.size()) {
        return best_index;
    }

    throw std::runtime_error("MetricDepthEstimator: could not identify depth output tensor");
}

MetricDepthEstimator::InputBuffer MetricDepthEstimator::prepareInput(
    size_t input_index,
    const cv::Mat& image,
    const cv::Size& input_size) const {
    InputBuffer input;
    input.type = input_types_[input_index];
    input.shape = input_shapes_[input_index];

    if (input_index == image_input_index_) {
        if (input.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("MetricDepthEstimator: image input must be float");
        }

        const bool nhwc_layout = isNhwcImageShape(input.shape);
        if (input.shape.size() == 4) {
            input.shape[0] = resolveDynamicDim(input.shape[0], 1);
            if (nhwc_layout) {
                input.shape[1] = input_size.height;
                input.shape[2] = input_size.width;
                input.shape[3] = resolveDynamicDim(input.shape[3], 3);
            } else {
                input.shape[1] = resolveDynamicDim(input.shape[1], 3);
                input.shape[2] = input_size.height;
                input.shape[3] = input_size.width;
            }
        } else {
            if (nhwc_layout) {
                input.shape[0] = input_size.height;
                input.shape[1] = input_size.width;
                input.shape[2] = resolveDynamicDim(input.shape[2], 3);
            } else {
                input.shape[0] = resolveDynamicDim(input.shape[0], 3);
                input.shape[1] = input_size.height;
                input.shape[2] = input_size.width;
            }
        }

        preprocessImage(image, input_size, nhwc_layout, input.float_data);
        return input;
    }

    for (auto& dim : input.shape) {
        dim = resolveDynamicDim(dim, 1);
    }
    const std::size_t element_count = checkedElementCount(input.shape);

    if (input.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        input.float_data = createFloatAuxiliaryInput(
            input_names_[input_index], element_count, image.size(), input_size);
        return input;
    }
    if (input.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        input.int64_data = createInt64AuxiliaryInput(
            input_names_[input_index], element_count, image.size(), input_size);
        return input;
    }

    throw std::runtime_error("MetricDepthEstimator: unsupported auxiliary input tensor type");
}

void MetricDepthEstimator::preprocessImage(const cv::Mat& image,
                                           const cv::Size& input_size,
                                           bool nhwc_layout,
                                           std::vector<float>& blob) const {
    cv::Mat bgr;
    if (image.channels() == 1) {
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    } else if (image.channels() == 3) {
        bgr = image;
    } else {
        cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
    }

    cv::Mat resized;
    cv::resize(bgr, resized, input_size, 0, 0, cv::INTER_LINEAR);

    cv::Mat float_img;
    resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    const float mean[] = {0.485f, 0.456f, 0.406f};
    const float stddev[] = {0.229f, 0.224f, 0.225f};

    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);

    blob.resize(static_cast<std::size_t>(input_size.height) *
                static_cast<std::size_t>(input_size.width) * 3U);

    if (nhwc_layout) {
        for (int y = 0; y < input_size.height; ++y) {
            const float* row_b = channels[0].ptr<float>(y);
            const float* row_g = channels[1].ptr<float>(y);
            const float* row_r = channels[2].ptr<float>(y);
            for (int x = 0; x < input_size.width; ++x) {
                const std::size_t offset =
                    (static_cast<std::size_t>(y) * input_size.width + x) * 3U;
                blob[offset + 0] = (row_r[x] - mean[2]) / stddev[2];
                blob[offset + 1] = (row_g[x] - mean[1]) / stddev[1];
                blob[offset + 2] = (row_b[x] - mean[0]) / stddev[0];
            }
        }
        return;
    }

    for (int c = 0; c < 3; ++c) {
        const int cv_channel = 2 - c;
        const float m = mean[cv_channel];
        const float s = stddev[cv_channel];

        for (int y = 0; y < input_size.height; ++y) {
            const float* row = channels[cv_channel].ptr<float>(y);
            for (int x = 0; x < input_size.width; ++x) {
                blob[static_cast<std::size_t>(c) * input_size.height * input_size.width +
                     static_cast<std::size_t>(y) * input_size.width + x] =
                    (row[x] - m) / s;
            }
        }
    }
}

std::vector<float> MetricDepthEstimator::createFloatAuxiliaryInput(
    const std::string& input_name,
    std::size_t element_count,
    const cv::Size& original_size,
    const cv::Size& input_size) const {
    const std::string lower_name = toLower(input_name);
    const float scale_x = original_size.width > 0
        ? static_cast<float>(input_size.width) / static_cast<float>(original_size.width)
        : 1.0f;
    const float scale_y = original_size.height > 0
        ? static_cast<float>(input_size.height) / static_cast<float>(original_size.height)
        : 1.0f;

    const float fx = camera_ ? static_cast<float>(camera_->fx_ * scale_x)
                             : static_cast<float>(input_size.width);
    const float fy = camera_ ? static_cast<float>(camera_->fy_ * scale_y)
                             : static_cast<float>(input_size.height);
    const float cx = camera_ ? static_cast<float>(camera_->cx_ * scale_x)
                             : 0.5f * static_cast<float>(input_size.width);
    const float cy = camera_ ? static_cast<float>(camera_->cy_ * scale_y)
                             : 0.5f * static_cast<float>(input_size.height);

    if (element_count == 9) {
        return {fx, 0.0f, cx,
                0.0f, fy, cy,
                0.0f, 0.0f, 1.0f};
    }
    if (element_count == 4) {
        if (looksLikeSizeName(lower_name) && !looksLikeIntrinsicName(lower_name)) {
            return {static_cast<float>(original_size.height), static_cast<float>(original_size.width),
                    static_cast<float>(input_size.height), static_cast<float>(input_size.width)};
        }
        return {fx, fy, cx, cy};
    }
    if (element_count == 2) {
        if (lower_name.find("focal") != std::string::npos) {
            return {fx, fy};
        }
        if (lower_name.find("principal") != std::string::npos ||
            lower_name.find("center") != std::string::npos) {
            return {cx, cy};
        }
        const cv::Size size_values =
            looksLikeOriginalSizeName(lower_name) ? original_size : input_size;
        return {static_cast<float>(size_values.height), static_cast<float>(size_values.width)};
    }

    throw std::runtime_error("MetricDepthEstimator: unsupported float auxiliary input " + input_name);
}

std::vector<int64_t> MetricDepthEstimator::createInt64AuxiliaryInput(
    const std::string& input_name,
    std::size_t element_count,
    const cv::Size& original_size,
    const cv::Size& input_size) const {
    const std::string lower_name = toLower(input_name);

    if (element_count == 2) {
        const cv::Size size_values =
            looksLikeOriginalSizeName(lower_name) ? original_size : input_size;
        return {static_cast<int64_t>(size_values.height), static_cast<int64_t>(size_values.width)};
    }
    if (element_count == 4 && looksLikeSizeName(lower_name)) {
        return {static_cast<int64_t>(original_size.height), static_cast<int64_t>(original_size.width),
                static_cast<int64_t>(input_size.height), static_cast<int64_t>(input_size.width)};
    }

    throw std::runtime_error("MetricDepthEstimator: unsupported int64 auxiliary input " + input_name);
}

cv::Mat MetricDepthEstimator::postprocess(const float* output_data,
                                          const std::vector<int64_t>& output_shape,
                                          const cv::Size& original_size) const {
    const cv::Size model_output_size = resolveOutputSize(output_shape);
    const std::size_t expected_count =
        static_cast<std::size_t>(model_output_size.width) * model_output_size.height;

    cv::Mat depth(model_output_size.height, model_output_size.width, CV_32FC1);
    std::memcpy(depth.data, output_data, expected_count * sizeof(float));

    for (int y = 0; y < depth.rows; ++y) {
        float* row = depth.ptr<float>(y);
        for (int x = 0; x < depth.cols; ++x) {
            if (!std::isfinite(row[x]) || row[x] <= 0.0f || row[x] > 1000.0f) {
                row[x] = 0.0f;
            }
        }
    }

    if (depth.size() == original_size) {
        return depth;
    }

    cv::Mat depth_orig;
    cv::resize(depth, depth_orig, original_size, 0, 0, cv::INTER_LINEAR);
    return depth_orig;
}

cv::Mat MetricDepthEstimator::estimate(const cv::Mat& image) {
    if (image.empty()) {
        return cv::Mat();
    }
    if (input_names_.empty()) {
        throw std::runtime_error("MetricDepthEstimator: no ONNX session has been loaded");
    }

    const cv::Size input_size = resolveInputSize(input_shapes_[image_input_index_], image.size());
    const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<InputBuffer> prepared_inputs;
    prepared_inputs.reserve(input_names_.size());
    for (size_t i = 0; i < input_names_.size(); ++i) {
        prepared_inputs.push_back(prepareInput(i, image, input_size));
    }

    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(prepared_inputs.size());
    for (const auto& prepared_input : prepared_inputs) {
        if (prepared_input.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info,
                const_cast<float*>(prepared_input.float_data.data()),
                prepared_input.float_data.size(),
                prepared_input.shape.data(),
                prepared_input.shape.size()));
            continue;
        }
        if (prepared_input.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info,
                const_cast<int64_t*>(prepared_input.int64_data.data()),
                prepared_input.int64_data.size(),
                prepared_input.shape.data(),
                prepared_input.shape.size()));
            continue;
        }
        throw std::runtime_error("MetricDepthEstimator: unsupported prepared input tensor type");
    }

    const char* depth_output_name = output_names_[depth_output_index_];
    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names_.data(),
        input_tensors.data(),
        input_tensors.size(),
        &depth_output_name,
        1);

    if (output_types_[depth_output_index_] != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("MetricDepthEstimator: depth output must be float");
    }

    const float* output_data = output_tensors[0].GetTensorData<float>();
    const auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    return postprocess(output_data, output_shape, image.size());
}

}

#endif // USE_DEPTH_DL
