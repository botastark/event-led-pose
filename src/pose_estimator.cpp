#include "pose_estimator.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <utility>

namespace event_led_pose {
namespace {

constexpr double RAD_TO_DEG = 57.2957795130823208768;

cv::Vec2d read_range(
    const cv::FileNode &node,
    const cv::Vec2d &fallback)
{
    if (node.empty() || node.size() != 2)
        return fallback;

    return cv::Vec2d(
        static_cast<double>(node[0]),
        static_cast<double>(node[1]));
}

std::array<cv::Point3d, 3> read_led_points(
    const cv::FileNode &node)
{
    if (node.empty())
        throw std::runtime_error("pose config: missing 'led_points_mm'");

    static constexpr const char *KEYS[3] = {
        "165", "366", "596"
    };

    std::array<cv::Point3d, 3> out{};

    for (int i = 0; i < 3; ++i) {
        const cv::FileNode p = node[KEYS[i]];

        if (p.empty() || (p.size() != 2 && p.size() != 3)) {
            throw std::runtime_error(
                std::string("pose config: led_points_mm.") +
                KEYS[i] + " must contain [x,y] or [x,y,z]");
        }

        out[i].x = static_cast<double>(p[0]);
        out[i].y = static_cast<double>(p[1]);
        out[i].z = p.size() == 3
            ? static_cast<double>(p[2])
            : 0.0;
    }

    return out;
}

bool within(double v, const cv::Vec2d &range) {
    return v >= range[0] && v <= range[1];
}

cv::Matx33d mat_to_matx33d(const cv::Mat &m) {
    cv::Matx33d out;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            out(r,c) = m.at<double>(r,c);
    return out;
}

} // namespace

PoseConfig load_pose_config(const std::string &path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);

    if (!fs.isOpened()) {
        throw std::runtime_error(
            "Could not open pose config: " + path);
    }

    PoseConfig cfg;

    const cv::FileNode camera = fs["camera"];
    if (camera.empty())
        throw std::runtime_error("pose config: missing 'camera'");

    const double fx = static_cast<double>(camera["fx"]);
    const double fy = static_cast<double>(camera["fy"]);
    const double cx = static_cast<double>(camera["cx"]);
    const double cy = static_cast<double>(camera["cy"]);

    if (!(fx > 0.0) || !(fy > 0.0))
        throw std::runtime_error("pose config: fx/fy must be > 0");

    cfg.camera_matrix = cv::Matx33d(
        fx, 0.0, cx,
        0.0, fy, cy,
        0.0, 0.0, 1.0);

    const cv::FileNode dist = camera["dist"];
    if (!dist.empty()) {
        cfg.dist_coeffs = cv::Mat::zeros(
            1,
            static_cast<int>(dist.size()),
            CV_64F);

        for (std::size_t i = 0; i < dist.size(); ++i) {
            cfg.dist_coeffs.at<double>(0, static_cast<int>(i)) =
                static_cast<double>(dist[static_cast<int>(i)]);
        }
    }
    else {
        cfg.dist_coeffs = cv::Mat::zeros(1, 5, CV_64F);
    }

    cfg.led_points_mm = read_led_points(fs["led_points_mm"]);

    const cv::FileNode quality = fs["quality"];
    if (!quality.empty()) {
        if (!quality["min_samples"].empty())
            cfg.min_samples =
                static_cast<std::uint32_t>(
                    static_cast<int>(quality["min_samples"]));

        if (!quality["max_p95_radius_px"].empty())
            cfg.max_p95_radius_px =
                static_cast<double>(quality["max_p95_radius_px"]);

        if (!quality["max_reprojection_rms_px"].empty())
            cfg.max_reprojection_rms_px =
                static_cast<double>(quality["max_reprojection_rms_px"]);
    }

    const cv::FileNode angles = fs["angle_limits_deg"];
    if (!angles.empty()) {
        cfg.roll_deg = read_range(angles["roll"], cfg.roll_deg);
        cfg.pitch_deg = read_range(angles["pitch"], cfg.pitch_deg);
        cfg.yaw_deg = read_range(angles["yaw"], cfg.yaw_deg);
    }

    const cv::FileNode continuity = fs["continuity"];
    if (!continuity.empty()) {
        if (!continuity["max_translation_jump_mm"].empty())
            cfg.max_translation_jump_mm =
                static_cast<double>(continuity["max_translation_jump_mm"]);

        if (!continuity["max_rotation_jump_deg"].empty())
            cfg.max_rotation_jump_deg =
                static_cast<double>(continuity["max_rotation_jump_deg"]);

        if (!continuity["translation_cost_per_mm"].empty())
            cfg.translation_cost_per_mm =
                static_cast<double>(continuity["translation_cost_per_mm"]);

        if (!continuity["rotation_cost_per_deg"].empty())
            cfg.rotation_cost_per_deg =
                static_cast<double>(continuity["rotation_cost_per_deg"]);
    }

    return cfg;
}

PoseEstimator::PoseEstimator(PoseConfig config)
    : config_(std::move(config))
{
    const cv::Point3d centroid =
        (config_.led_points_mm[0] +
         config_.led_points_mm[1] +
         config_.led_points_mm[2]) * (1.0 / 3.0);

    for (int i = 0; i < 3; ++i) {
        object_points_centered_[i] =
            config_.led_points_mm[i] - centroid;
    }

    const cv::Vec3d a(
        object_points_centered_[1].x - object_points_centered_[0].x,
        object_points_centered_[1].y - object_points_centered_[0].y,
        object_points_centered_[1].z - object_points_centered_[0].z);

    const cv::Vec3d b(
        object_points_centered_[2].x - object_points_centered_[0].x,
        object_points_centered_[2].y - object_points_centered_[0].y,
        object_points_centered_[2].z - object_points_centered_[0].z);

    const double area2 = cv::norm(a.cross(b));
    if (area2 < 1e-6)
        throw std::runtime_error("pose config: LED points are collinear");
}

void PoseEstimator::reset() noexcept {
    have_previous_ = false;
    previous_tvec_ = cv::Vec3d(0.0, 0.0, 0.0);
    previous_R_ = cv::Matx33d::eye();
}

bool PoseEstimator::centers_pass_quality_gate(
    const CenterSnapshot &centers) const
{
    for (std::size_t i = 0; i < 3; ++i) {
        const auto &s = centers.frequency[i];

        if (!s.valid)
            return false;

        if (s.sample_count < config_.min_samples)
            return false;

        if (config_.max_p95_radius_px > 0.0 &&
            s.p95_radius > config_.max_p95_radius_px)
        {
            return false;
        }
    }

    return true;
}

cv::Vec3d PoseEstimator::rotation_to_rpy_deg(
    const cv::Matx33d &R)
{
    // ZYX decomposition: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    const double sy = std::sqrt(
        R(0,0) * R(0,0) +
        R(1,0) * R(1,0));

    const bool singular = sy < 1e-9;

    double roll;
    double pitch;
    double yaw;

    if (!singular) {
        roll  = std::atan2(R(2,1), R(2,2));
        pitch = std::atan2(-R(2,0), sy);
        yaw   = std::atan2(R(1,0), R(0,0));
    }
    else {
        roll  = std::atan2(-R(1,2), R(1,1));
        pitch = std::atan2(-R(2,0), sy);
        yaw   = 0.0;
    }

    return cv::Vec3d(
        roll * RAD_TO_DEG,
        pitch * RAD_TO_DEG,
        yaw * RAD_TO_DEG);
}

double PoseEstimator::rotation_distance_deg(
    const cv::Matx33d &a,
    const cv::Matx33d &b)
{
    const cv::Matx33d d = a * b.t();

    double cos_angle =
        (d(0,0) + d(1,1) + d(2,2) - 1.0) * 0.5;

    cos_angle = std::clamp(cos_angle, -1.0, 1.0);

    return std::acos(cos_angle) * RAD_TO_DEG;
}

PoseResult PoseEstimator::estimate(
    const CenterSnapshot &centers)
{
    PoseResult result;
    result.timestamp_us = centers.newest_t;

    if (!centers_pass_quality_gate(centers))
        return result;

    std::vector<cv::Point3d> object_points(
        object_points_centered_.begin(),
        object_points_centered_.end());

    std::vector<cv::Point2d> image_points;
    image_points.reserve(3);

    for (std::size_t i = 0; i < 3; ++i) {
        image_points.emplace_back(
            centers.frequency[i].x,
            centers.frequency[i].y);
    }

    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
// SOLVEPNP_SQPNP
    const int solutions = cv::solveP3P(
        object_points,
        image_points,
        cv::Mat(config_.camera_matrix),
        config_.dist_coeffs,
        rvecs,
        tvecs,
        cv::SOLVEPNP_P3P);

    // const int solutions = cv::solvePnPGeneric(
    //     object_points,
    //     image_points,
    //     cv::Mat(config_.camera_matrix),
    //     config_.dist_coeffs,
    //     rvecs,
    //     tvecs,
    //     false,
    //     cv::SOLVEPNP_SQPNP);

    result.candidate_count = solutions;

    if (solutions <= 0)
        return result;

    double best_cost = std::numeric_limits<double>::infinity();
    int accepted = 0;

    PoseResult best;
    best.timestamp_us = centers.newest_t;
    best.candidate_count = solutions;

    for (int i = 0; i < solutions; ++i) {
        cv::Mat rvec64;
        cv::Mat tvec64;
        rvecs[i].convertTo(rvec64, CV_64F);
        tvecs[i].convertTo(tvec64, CV_64F);

        const cv::Vec3d rvec(
            rvec64.at<double>(0),
            rvec64.at<double>(1),
            rvec64.at<double>(2));

        const cv::Vec3d tvec(
            tvec64.at<double>(0),
            tvec64.at<double>(1),
            tvec64.at<double>(2));

        cv::Mat Rmat;
        cv::Rodrigues(rvec, Rmat);
        const cv::Matx33d R = mat_to_matx33d(Rmat);

        auto &candidate = best.candidates.at(static_cast<std::size_t>(i));
        candidate.position_mm = tvec;

        // All physical LEDs must lie in front of the camera.
        bool positive_depth = true;
        for (const auto &p : object_points_centered_) {
            const cv::Vec3d pc =
                R * cv::Vec3d(p.x, p.y, p.z) + tvec;

            if (pc[2] <= 0.0) {
                positive_depth = false;
                break;
            }
        }

        if (!positive_depth)
            candidate.rejection_flags |= RejectDepth;
        // Preserve the uploaded estimator's raw marker-to-camera RPY.
        const cv::Vec3d rpy = rotation_to_rpy_deg(R);
        candidate.rpy_deg = rpy;
        if (!within(rpy[0], config_.roll_deg))
            candidate.rejection_flags |= RejectRoll;
        if (!within(rpy[1], config_.pitch_deg))
            candidate.rejection_flags |= RejectPitch;
        if (!within(rpy[2], config_.yaw_deg))
            candidate.rejection_flags |= RejectYaw;

        std::vector<cv::Point2d> projected;
        cv::projectPoints(
            object_points,
            rvec,
            tvec,
            cv::Mat(config_.camera_matrix),
            config_.dist_coeffs,
            projected);

        double sum_sq = 0.0;
        for (std::size_t k = 0; k < 3; ++k) {
            const cv::Point2d d = projected[k] - image_points[k];
            sum_sq += d.x * d.x + d.y * d.y;
        }

        const double reproj_rms = std::sqrt(sum_sq / 3.0);
        candidate.reprojection_rms_px = reproj_rms;
        if (reproj_rms > config_.max_reprojection_rms_px)
            candidate.rejection_flags |= RejectFit;

        double translation_jump = 0.0;
        double rotation_jump = 0.0;

        if (have_previous_) {
            translation_jump = cv::norm(tvec - previous_tvec_);
            rotation_jump = rotation_distance_deg(R, previous_R_);

            if (config_.max_translation_jump_mm > 0.0 &&
                translation_jump > config_.max_translation_jump_mm)
            {
                candidate.rejection_flags |= RejectTranslation;
            }

            if (config_.max_rotation_jump_deg > 0.0 &&
                rotation_jump > config_.max_rotation_jump_deg)
            {
                candidate.rejection_flags |= RejectRotation;
            }
        }

        candidate.translation_jump_mm = translation_jump;
        candidate.rotation_jump_deg = rotation_jump;
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(tvec[axis]) || !std::isfinite(rpy[axis]))
                candidate.rejection_flags |= RejectNonfinite;
        }
        if (!std::isfinite(reproj_rms) ||
            !std::isfinite(translation_jump) || !std::isfinite(rotation_jump))
            candidate.rejection_flags |= RejectNonfinite;

        // Retain diagnostics even when one or more gates reject this pose.
        if (candidate.rejection_flags != 0)
            continue;

        ++accepted;

        const double cost =
            reproj_rms +
            config_.translation_cost_per_mm * translation_jump +
            config_.rotation_cost_per_deg * rotation_jump;

        if (cost >= best_cost)
            continue;

        best_cost = cost;

        best.valid = true;
        best.selected_candidate = i;
        best.position_mm = tvec;
        best.rvec = rvec;
        best.rpy_deg = rpy;
        best.reprojection_rms_px = reproj_rms;
        best.translation_jump_mm = translation_jump;
        best.rotation_jump_deg = rotation_jump;
    }

    best.accepted_candidate_count = accepted;

    if (!best.valid)
        return best;

    cv::Mat Rmat;
    cv::Rodrigues(best.rvec, Rmat);

    previous_R_ = mat_to_matx33d(Rmat);
    previous_tvec_ = best.position_mm;
    have_previous_ = true;

    return best;
}

} // namespace event_led_pose
