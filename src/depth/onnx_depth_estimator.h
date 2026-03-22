#pragma once
#ifdef USE_DEPTH_DL

#include "depth/depth_estimator.h"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

namespace svslam {

class OnnxDepthEstimator : public DepthEstimator {
public:
    explicit OnnxDepthEstimator(const std::string& model_path);
    ~OnnxDepthEstimator() override = default;

    cv::Mat estimate(const cv::Mat& image) override;
    bool isMetric() const override { return false; }

private:
    void preprocess(const cv::Mat& image, std::vector<float>& blob);
    cv::Mat postprocess(const std::vector<float>& output, int orig_h, int orig_w);

    Ort::Env env_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    static constexpr int kInputH = 518;
    static constexpr int kInputW = 518;

    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<Ort::AllocatedStringPtr> input_name_ptrs_;
    std::vector<Ort::AllocatedStringPtr> output_name_ptrs_;
};

}

#endif // USE_DEPTH_DL
