#pragma once

#include "center_worker.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <cstdint>
#include <string>

namespace event_led_pose {

enum PoseRejection : unsigned {
    RejectDepth = 1u, RejectRoll = 2u, RejectPitch = 4u,
    RejectYaw = 8u, RejectFit = 16u, RejectTranslation = 32u,
    RejectRotation = 64u, RejectNonfinite = 128u
};

struct PoseCandidate {
    cv::Vec3d position_mm{0.0, 0.0, 0.0};
    cv::Vec3d rpy_deg{0.0, 0.0, 0.0};
    double reprojection_rms_px = 0.0;
    double translation_jump_mm = 0.0;
    double rotation_jump_deg = 0.0;
    unsigned rejection_flags = 0;
};

struct PoseResult {
    bool valid = false;

    // Triangle-centroid pose in the camera frame.
    cv::Vec3d position_mm{0.0, 0.0, 0.0};
    cv::Vec3d rvec{0.0, 0.0, 0.0};

    // ZYX Euler decomposition of the marker->camera rotation:
    // roll around camera X, pitch around camera Y, yaw around camera Z.
    cv::Vec3d rpy_deg{0.0, 0.0, 0.0};

    double reprojection_rms_px = 0.0;
    double translation_jump_mm = 0.0;
    double rotation_jump_deg = 0.0;

    int candidate_count = 0;
    int accepted_candidate_count = 0;
    // P3P returns at most four solutions. Indices are frame-local,
    // not persistent identities across frames.
    std::array<PoseCandidate, 4> candidates{};
    int selected_candidate = -1;

    std::uint32_t timestamp_us = 0;
};

struct PoseConfig {
    cv::Matx33d camera_matrix = cv::Matx33d::eye();
    cv::Mat dist_coeffs;

    // Input coordinates may use any origin. load_pose_config() recenters
    // them so the marker-frame origin is the triangle centroid.
    std::array<cv::Point3d, 3> led_points_mm{};

    // Per-LED center quality gates, in frequency order 165/366/596.
    std::uint32_t min_samples = 8;
    double max_p95_radius_px = 80.0;

    // Solution validity / ambiguity constraints.
    double max_reprojection_rms_px = 5.0;

    cv::Vec2d roll_deg{-70.0, 70.0};
    cv::Vec2d pitch_deg{-70.0, 70.0};
    cv::Vec2d yaw_deg{-180.0, 180.0};

    // Temporal continuity. Set <= 0 to disable a gate.
    double max_translation_jump_mm = 250.0;
    double max_rotation_jump_deg = 45.0;

    // Candidate ranking after hard gates.
    double translation_cost_per_mm = 0.01;
    double rotation_cost_per_deg = 0.05;
};

PoseConfig load_pose_config(const std::string &path);

class PoseEstimator {
public:
    explicit PoseEstimator(PoseConfig config);

    const PoseConfig &config() const noexcept { return config_; }

    // Uses frequency order:
    //   0 -> 165 Hz green
    //   1 -> 366 Hz yellow
    //   2 -> 596 Hz blue
    PoseResult estimate(const CenterSnapshot &centers);

    void reset() noexcept;

private:
    PoseConfig config_;

    std::array<cv::Point3d, 3> object_points_centered_{};

    bool have_previous_ = false;
    cv::Vec3d previous_tvec_{0.0, 0.0, 0.0};
    cv::Matx33d previous_R_ = cv::Matx33d::eye();

    static cv::Vec3d rotation_to_rpy_deg(const cv::Matx33d &R);
    static double rotation_distance_deg(
        const cv::Matx33d &a,
        const cv::Matx33d &b);

    bool centers_pass_quality_gate(const CenterSnapshot &centers) const;
};

} // namespace event_led_pose
