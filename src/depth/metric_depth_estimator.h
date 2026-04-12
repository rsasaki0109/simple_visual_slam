#pragma once
#ifdef USE_DEPTH_DL

#include "core/camera.h"
#include "depth/depth_estimator.h"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

namespace svslam {

class MetricDepthEstimator : public DepthEstimator {
public:
    explicit MetricDepthEstimator(const std::string& model_path, Camera::Ptr camera = nullptr);
    explicit MetricDepthEstimator(Camera::Ptr camera = nullptr);
    ~MetricDepthEstimator() override = default;

    cv::Mat estimate(const cv::Mat& image) override;
    bool isMetric() const override { return true; }

    static cv::Size resolveInputSize(const std::vector<int64_t>& input_shape,
                                     const cv::Size& image_size);
    static cv::Size resolveOutputSize(const std::vector<int64_t>& output_shape);

private:
    struct InputBuffer {
        std::vector<int64_t> shape;
        ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        std::vector<float> float_data;
        std::vector<int64_t> int64_data;
    };

    static bool isNhwcImageShape(const std::vector<int64_t>& shape);

    size_t findImageInputIndex() const;
    size_t findDepthOutputIndex() const;
    InputBuffer prepareInput(size_t input_index,
                             const cv::Mat& image,
                             const cv::Size& input_size) const;
    void preprocessImage(const cv::Mat& image,
                         const cv::Size& input_size,
                         bool nhwc_layout,
                         std::vector<float>& blob) const;
    std::vector<float> createFloatAuxiliaryInput(const std::string& input_name,
                                                 std::size_t element_count,
                                                 const cv::Size& original_size,
                                                 const cv::Size& input_size) const;
    std::vector<int64_t> createInt64AuxiliaryInput(const std::string& input_name,
                                                   std::size_t element_count,
                                                   const cv::Size& original_size,
                                                   const cv::Size& input_size) const;
    cv::Mat postprocess(const float* output_data,
                        const std::vector<int64_t>& output_shape,
                        const cv::Size& original_size) const;

    Camera::Ptr camera_;
    Ort::Env env_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    size_t image_input_index_ = 0;
    size_t depth_output_index_ = 0;

    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<Ort::AllocatedStringPtr> input_name_ptrs_;
    std::vector<Ort::AllocatedStringPtr> output_name_ptrs_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
    std::vector<ONNXTensorElementDataType> input_types_;
    std::vector<ONNXTensorElementDataType> output_types_;
};

}

#endif // USE_DEPTH_DL
