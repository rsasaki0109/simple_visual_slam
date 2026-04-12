#ifdef USE_DEPTH_DL

#include <gtest/gtest.h>

#include "depth/metric_depth_estimator.h"

using namespace svslam;

TEST(MetricDepthEstimatorTest, ReportsMetricDepthForNullSessionInstance) {
    MetricDepthEstimator estimator;
    EXPECT_TRUE(estimator.isMetric());
}

TEST(MetricDepthEstimatorTest, ResolvesStaticNchwInputSize) {
    const cv::Size input_size =
        MetricDepthEstimator::resolveInputSize({1, 3, 616, 1064}, cv::Size(640, 480));

    EXPECT_EQ(input_size.width, 1064);
    EXPECT_EQ(input_size.height, 616);
}

TEST(MetricDepthEstimatorTest, ResolvesDynamicNhwcInputSizeFromImageShape) {
    const cv::Size input_size =
        MetricDepthEstimator::resolveInputSize({1, -1, -1, 3}, cv::Size(640, 480));

    EXPECT_EQ(input_size.width, 640);
    EXPECT_EQ(input_size.height, 480);
}

TEST(MetricDepthEstimatorTest, ResolvesSingleChannelDepthOutputShapes) {
    const cv::Size nchw_output = MetricDepthEstimator::resolveOutputSize({1, 1, 616, 1064});
    EXPECT_EQ(nchw_output.width, 1064);
    EXPECT_EQ(nchw_output.height, 616);

    const cv::Size nhw_output = MetricDepthEstimator::resolveOutputSize({1, 480, 640});
    EXPECT_EQ(nhw_output.width, 640);
    EXPECT_EQ(nhw_output.height, 480);
}

#endif // USE_DEPTH_DL
