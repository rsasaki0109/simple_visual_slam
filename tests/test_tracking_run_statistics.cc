#include <gtest/gtest.h>

#include "tracking/tracking.h"

using namespace svslam;

TEST(TrackingRunStatisticsTest, DefaultsAreZero) {
    Tracking t;
    const TrackingRunStatistics s = t.runStatistics();
    EXPECT_EQ(s.reloc_attempts, 0u);
    EXPECT_EQ(s.reloc_successes, 0u);
    EXPECT_EQ(s.frames_tracking_lost, 0u);
    EXPECT_EQ(s.reinit_successes, 0u);
}
