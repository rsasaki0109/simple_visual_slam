#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "core/heuristic_reference_keyframe_policy.h"
#include "experiments/reference_keyframe/pipeline_reference_keyframe_policy.h"
#include "experiments/reference_keyframe/score_reference_keyframe_policy.h"

namespace svslam {
namespace {

struct Scenario {
    std::string name;
    std::string mode;
    ReferenceKeyframePolicyInput input;
    ReferenceKeyframeAction expected_action = ReferenceKeyframeAction::KeepCurrentReference;
    std::string note;
};

struct PolicyMetrics {
    std::string name;
    std::string philosophy;
    double accuracy = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double promote_rate = 0.0;
    double mean_confidence = 0.0;
    double mean_eval_ns = 0.0;
    int true_positive = 0;
    int false_positive = 0;
    int true_negative = 0;
    int false_negative = 0;
};

std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> parts;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
        parts.push_back(cell);
    }
    return parts;
}

bool parseBool(const std::string& value) {
    return value == "1" || value == "true" || value == "TRUE";
}

ReferenceKeyframeAction parseAction(const std::string& value) {
    return value == "promote"
        ? ReferenceKeyframeAction::PromoteNewReference
        : ReferenceKeyframeAction::KeepCurrentReference;
}

std::vector<Scenario> loadScenarios(const std::string& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open scenario file: " + path);
    }

    std::vector<Scenario> scenarios;
    std::string line;
    bool first_line = true;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        if (first_line) {
            first_line = false;
            continue;
        }

        const auto cells = splitCsv(line);
        if (cells.size() != 11) {
            throw std::runtime_error("Unexpected scenario row: " + line);
        }

        Scenario scenario;
        scenario.name = cells[0];
        scenario.mode = cells[1];
        scenario.input.tracked_features = std::stoi(cells[2]);
        scenario.input.detected_keypoints = std::stoi(cells[3]);
        scenario.input.candidate_landmarks = std::stoi(cells[4]);
        scenario.input.frames_since_reference = std::stoi(cells[5]);
        scenario.input.lost_frames = std::stoi(cells[6]);
        scenario.input.has_depth = parseBool(cells[7]);
        scenario.input.has_accel = parseBool(cells[8]);
        scenario.expected_action = parseAction(cells[9]);
        scenario.note = cells[10];
        scenarios.push_back(scenario);
    }
    return scenarios;
}

PolicyMetrics evaluatePolicy(
    const ReferenceKeyframePolicy& policy,
    const std::vector<Scenario>& scenarios,
    const std::string& decisions_path) {
    std::ofstream decisions(decisions_path, std::ios::app);
    if (!decisions.is_open()) {
        throw std::runtime_error("Failed to open decisions file: " + decisions_path);
    }

    constexpr int kBenchmarkIterations = 200000;
    volatile double sink = 0.0;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < kBenchmarkIterations; ++i) {
        for (const auto& scenario : scenarios) {
            const auto decision = policy.evaluate(scenario.input);
            sink += decision.promoteNewReference() ? 1.0 : 0.0;
            sink += decision.confidence;
        }
    }
    auto finish = std::chrono::steady_clock::now();
    (void)sink;

    PolicyMetrics metrics;
    metrics.name = policy.name();
    metrics.philosophy = policy.philosophy();
    metrics.mean_eval_ns = std::chrono::duration<double, std::nano>(finish - start).count() /
                           static_cast<double>(kBenchmarkIterations * std::max<size_t>(1, scenarios.size()));

    int promote_count = 0;
    double confidence_total = 0.0;
    for (const auto& scenario : scenarios) {
        const auto decision = policy.evaluate(scenario.input);
        const bool predicted_promote = decision.promoteNewReference();
        const bool expected_promote = scenario.expected_action == ReferenceKeyframeAction::PromoteNewReference;
        const bool match = predicted_promote == expected_promote;

        promote_count += predicted_promote ? 1 : 0;
        confidence_total += decision.confidence;

        if (predicted_promote && expected_promote) metrics.true_positive++;
        if (predicted_promote && !expected_promote) metrics.false_positive++;
        if (!predicted_promote && !expected_promote) metrics.true_negative++;
        if (!predicted_promote && expected_promote) metrics.false_negative++;

        decisions << metrics.name << ','
                  << scenario.name << ','
                  << toString(scenario.expected_action) << ','
                  << toString(decision.action) << ','
                  << std::fixed << std::setprecision(4) << decision.confidence << ','
                  << (match ? "1" : "0") << ','
                  << decision.reason << '\n';
    }

    const double total = static_cast<double>(std::max<size_t>(1, scenarios.size()));
    metrics.accuracy = static_cast<double>(metrics.true_positive + metrics.true_negative) / total;
    metrics.precision = metrics.true_positive + metrics.false_positive == 0
        ? 0.0
        : static_cast<double>(metrics.true_positive) /
              static_cast<double>(metrics.true_positive + metrics.false_positive);
    metrics.recall = metrics.true_positive + metrics.false_negative == 0
        ? 0.0
        : static_cast<double>(metrics.true_positive) /
              static_cast<double>(metrics.true_positive + metrics.false_negative);
    metrics.promote_rate = static_cast<double>(promote_count) / total;
    metrics.mean_confidence = confidence_total / total;
    return metrics;
}

void writeMetrics(const std::string& metrics_path, const std::vector<PolicyMetrics>& metrics) {
    std::ofstream output(metrics_path);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to open metrics file: " + metrics_path);
    }

    output << "policy,philosophy,accuracy,precision,recall,promote_rate,mean_confidence,mean_eval_ns,tp,fp,tn,fn\n";
    for (const auto& metric : metrics) {
        output << metric.name << ','
               << metric.philosophy << ','
               << std::fixed << std::setprecision(6)
               << metric.accuracy << ','
               << metric.precision << ','
               << metric.recall << ','
               << metric.promote_rate << ','
               << metric.mean_confidence << ','
               << metric.mean_eval_ns << ','
               << metric.true_positive << ','
               << metric.false_positive << ','
               << metric.true_negative << ','
               << metric.false_negative << '\n';
    }
}

void writeDecisionHeader(const std::string& decisions_path) {
    std::ofstream output(decisions_path);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to open decisions file: " + decisions_path);
    }
    output << "policy,scenario,expected_action,actual_action,confidence,match,reason\n";
}

}  // namespace
}  // namespace svslam

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <scenarios.csv> <metrics.csv> <decisions.csv>" << std::endl;
        return 1;
    }

    try {
        const std::string scenario_path = argv[1];
        const std::string metrics_path = argv[2];
        const std::string decisions_path = argv[3];

        const auto scenarios = svslam::loadScenarios(scenario_path);
        svslam::writeDecisionHeader(decisions_path);

        std::vector<std::unique_ptr<svslam::ReferenceKeyframePolicy>> policies;
        policies.push_back(std::make_unique<svslam::HeuristicReferenceKeyframePolicy>());
        policies.push_back(std::make_unique<svslam::ScoreReferenceKeyframePolicy>());
        policies.push_back(std::make_unique<svslam::PipelineReferenceKeyframePolicy>());

        std::vector<svslam::PolicyMetrics> metrics;
        metrics.reserve(policies.size());
        for (const auto& policy : policies) {
            metrics.push_back(svslam::evaluatePolicy(*policy, scenarios, decisions_path));
        }

        svslam::writeMetrics(metrics_path, metrics);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 2;
    }
}
