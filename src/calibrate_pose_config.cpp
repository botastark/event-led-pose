#include <metavision/sdk/stream/camera.h>
#include <metavision/sdk/base/events/event_cd.h>

#include <opencv2/opencv.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr const char *DEFAULT_SERIAL = "00050946";

constexpr int COLUMNS = 9;
constexpr int ROWS    = 6;

// Only intrinsics are used by the tracker, so the absolute square scale
// does not affect fx/fy/cx/cy/dist. Keep the measured board pitch here.
constexpr float SQUARE_SIZE_M = 0.03883f;

constexpr int MIN_SAMPLES = 20;
constexpr int TARGET_SAMPLES = 35;

// Event-image construction.
constexpr int LIVE_PERSISTENCE_MS = 12;
constexpr int DETECTION_ACCUMULATION_MS = 120;
constexpr std::uint8_t DETECTION_EVENT_INCREMENT = 24;
constexpr double DETECT_SCALE = 1.0;

// Robust view rejection.
constexpr double MAX_VIEW_ERROR_PX = 0.80;
constexpr int MAX_REJECTION_ROUNDS = 100;

constexpr std::size_t MAX_EVENTS = 1'500'000;

struct PackedEvent {
    std::uint16_t x;
    std::uint16_t y;
    std::uint8_t p;
};

struct CalibrationResult {
    cv::Mat K;
    cv::Mat D;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    std::vector<float> per_view_errors;
    double rms = 0.0;
    double mean_reprojection_error = 0.0;
};

struct PoseTemplate {
    std::array<cv::Point3d, 3> led_points{};

    int min_samples = 5;
    double max_p95_radius_px = 0.0;
    double max_reprojection_rms_px = 3.0;

    cv::Vec2d roll_deg{-180.0, 180.0};
    cv::Vec2d pitch_deg{-180.0, 180.0};
    cv::Vec2d yaw_deg{-180.0, 180.0};

    double max_translation_jump_mm = 0.0;
    double max_rotation_jump_deg = 0.0;
    double translation_cost_per_mm = 0.0;
    double rotation_cost_per_deg = 0.0;
};

struct Options {
    std::string serial = DEFAULT_SERIAL;
    std::string pose_template;
    std::string output = "pose_calibrated.yml";
};

std::atomic<bool> g_running{true};

std::mutex g_events_mutex;
std::vector<PackedEvent> g_events;

std::mutex g_detect_input_mutex;
cv::Mat g_newest_detect_image;
std::atomic<bool> g_detect_image_ready{false};

std::mutex g_result_mutex;
std::vector<cv::Point2f> g_latest_corners;
bool g_found = false;
double g_detect_ms = 0.0;
std::chrono::steady_clock::time_point g_result_time;

Options parse_args(int argc, char **argv) {
    Options o;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];

        if (a == "--pose-template") {
            if (++i >= argc)
                throw std::runtime_error("--pose-template needs FILE.yml");
            o.pose_template = argv[i];
        }
        else if (a == "--output") {
            if (++i >= argc)
                throw std::runtime_error("--output needs FILE.yml");
            o.output = argv[i];
        }
        else if (a == "--serial") {
            if (++i >= argc)
                throw std::runtime_error("--serial needs camera serial");
            o.serial = argv[i];
        }
        else if (a == "--help" || a == "-h") {
            std::cout
                << "Usage:\n"
                << "  calibrate_pose_config --pose-template pose.yml "
                   "[--output pose_calibrated.yml] [--serial SERIAL]\n\n"
                << "Controls:\n"
                << "  SPACE  capture current checkerboard view\n"
                << "  C      calibrate and write tracker-ready pose YAML\n"
                << "  X      clear captured samples\n"
                << "  Q/Esc  quit\n";
            std::exit(0);
        }
        else {
            throw std::runtime_error("Unknown argument: " + a);
        }
    }

    if (o.pose_template.empty())
        throw std::runtime_error(
            "--pose-template is required so LED geometry and pose gates "
            "can be preserved in the output config");

    return o;
}

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

PoseTemplate load_pose_template(const std::string &path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);

    if (!fs.isOpened())
        throw std::runtime_error("Could not open pose template: " + path);

    PoseTemplate out;

    const cv::FileNode led = fs["led_points_mm"];
    if (led.empty())
        throw std::runtime_error("pose template: missing led_points_mm");

    static constexpr const char *KEYS[3] = {"165", "366", "596"};

    for (int i = 0; i < 3; ++i) {
        const cv::FileNode p = led[KEYS[i]];

        if (p.empty() || (p.size() != 2 && p.size() != 3)) {
            throw std::runtime_error(
                std::string("pose template: led_points_mm.") +
                KEYS[i] + " must contain [x,y] or [x,y,z]");
        }

        out.led_points[i].x = static_cast<double>(p[0]);
        out.led_points[i].y = static_cast<double>(p[1]);
        out.led_points[i].z =
            p.size() == 3 ? static_cast<double>(p[2]) : 0.0;
    }

    const cv::FileNode q = fs["quality"];
    if (!q.empty()) {
        if (!q["min_samples"].empty())
            out.min_samples = static_cast<int>(q["min_samples"]);
        if (!q["max_p95_radius_px"].empty())
            out.max_p95_radius_px =
                static_cast<double>(q["max_p95_radius_px"]);
        if (!q["max_reprojection_rms_px"].empty())
            out.max_reprojection_rms_px =
                static_cast<double>(q["max_reprojection_rms_px"]);
    }

    const cv::FileNode angles = fs["angle_limits_deg"];
    if (!angles.empty()) {
        out.roll_deg = read_range(angles["roll"], out.roll_deg);
        out.pitch_deg = read_range(angles["pitch"], out.pitch_deg);
        out.yaw_deg = read_range(angles["yaw"], out.yaw_deg);
    }

    const cv::FileNode continuity = fs["continuity"];
    if (!continuity.empty()) {
        if (!continuity["max_translation_jump_mm"].empty())
            out.max_translation_jump_mm =
                static_cast<double>(
                    continuity["max_translation_jump_mm"]);

        if (!continuity["max_rotation_jump_deg"].empty())
            out.max_rotation_jump_deg =
                static_cast<double>(
                    continuity["max_rotation_jump_deg"]);

        if (!continuity["translation_cost_per_mm"].empty())
            out.translation_cost_per_mm =
                static_cast<double>(
                    continuity["translation_cost_per_mm"]);

        if (!continuity["rotation_cost_per_deg"].empty())
            out.rotation_cost_per_deg =
                static_cast<double>(
                    continuity["rotation_cost_per_deg"]);
    }

    return out;
}

std::vector<cv::Point3f> make_object_points() {
    std::vector<cv::Point3f> pts;
    pts.reserve(COLUMNS * ROWS);

    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLUMNS; ++c) {
            pts.emplace_back(
                c * SQUARE_SIZE_M,
                r * SQUARE_SIZE_M,
                0.0f);
        }
    }

    return pts;
}

CalibrationResult calibrate_once(
    const std::vector<std::vector<cv::Point2f>> &image_points,
    const cv::Size image_size)
{
    CalibrationResult result;

    const auto obj = make_object_points();
    std::vector<std::vector<cv::Point3f>> object_points(
        image_points.size(), obj);

    result.K = cv::Mat::eye(3, 3, CV_64F);

    // Standard 5-parameter Brown-Conrady model, matching current tracker.
    result.D = cv::Mat::zeros(1, 5, CV_64F);

    result.rms = cv::calibrateCamera(
        object_points,
        image_points,
        image_size,
        result.K,
        result.D,
        result.rvecs,
        result.tvecs,
        0);

    double total_sq_error = 0.0;
    std::size_t total_points = 0;

    result.per_view_errors.reserve(image_points.size());

    for (std::size_t i = 0; i < image_points.size(); ++i) {
        std::vector<cv::Point2f> projected;

        cv::projectPoints(
            object_points[i],
            result.rvecs[i],
            result.tvecs[i],
            result.K,
            result.D,
            projected);

        const double err =
            cv::norm(image_points[i], projected, cv::NORM_L2);

        const double per_view =
            std::sqrt(
                (err * err) /
                static_cast<double>(projected.size()));

        result.per_view_errors.push_back(
            static_cast<float>(per_view));

        total_sq_error += err * err;
        total_points += projected.size();
    }

    result.mean_reprojection_error =
        std::sqrt(
            total_sq_error /
            static_cast<double>(total_points));

    return result;
}

void write_pose_config(
    const std::string &path,
    const CalibrationResult &result,
    const PoseTemplate &pose,
    const std::string &serial,
    const cv::Size image_size,
    int original_sample_count,
    int accepted_sample_count,
    const std::vector<int> &accepted_indices,
    const std::vector<int> &rejected_indices)
{
    std::ofstream out(path);

    if (!out)
        throw std::runtime_error("Could not open output file: " + path);

    out << std::setprecision(17);

    out << "%YAML:1.0\n";
    out << "---\n";

    // Exact schema consumed by load_pose_config().
    out << "camera:\n";
    out << "   fx: " << result.K.at<double>(0, 0) << "\n";
    out << "   fy: " << result.K.at<double>(1, 1) << "\n";
    out << "   cx: " << result.K.at<double>(0, 2) << "\n";
    out << "   cy: " << result.K.at<double>(1, 2) << "\n";
    out << "   dist: [ ";

    for (int i = 0; i < result.D.cols; ++i) {
        if (i)
            out << ", ";
        out << result.D.at<double>(0, i);
    }

    out << " ]\n";

    // Quote frequency labels explicitly. OpenCV FileStorage's writer can
    // reject map keys that begin with digits, so emit these keys manually.
    out << "led_points_mm:\n";

    static constexpr const char *KEYS[3] = {"165", "366", "596"};

    for (int i = 0; i < 3; ++i) {
        const auto &p = pose.led_points[i];

        out << "   \"" << KEYS[i] << "\": [ "
            << p.x << ", "
            << p.y << ", "
            << p.z << " ]\n";
    }

    out << "quality:\n";
    out << "   min_samples: " << pose.min_samples << "\n";
    out << "   max_p95_radius_px: "
        << pose.max_p95_radius_px << "\n";
    out << "   max_reprojection_rms_px: "
        << pose.max_reprojection_rms_px << "\n";

    out << "angle_limits_deg:\n";
    out << "   roll: [ "
        << pose.roll_deg[0] << ", "
        << pose.roll_deg[1] << " ]\n";
    out << "   pitch: [ "
        << pose.pitch_deg[0] << ", "
        << pose.pitch_deg[1] << " ]\n";
    out << "   yaw: [ "
        << pose.yaw_deg[0] << ", "
        << pose.yaw_deg[1] << " ]\n";

    out << "continuity:\n";
    out << "   max_translation_jump_mm: "
        << pose.max_translation_jump_mm << "\n";
    out << "   max_rotation_jump_deg: "
        << pose.max_rotation_jump_deg << "\n";
    out << "   translation_cost_per_mm: "
        << pose.translation_cost_per_mm << "\n";
    out << "   rotation_cost_per_deg: "
        << pose.rotation_cost_per_deg << "\n";

    out << "checkerboard_calibration:\n";
    out << "   camera_serial: \"" << serial << "\"\n";
    out << "   image_width: " << image_size.width << "\n";
    out << "   image_height: " << image_size.height << "\n";
    out << "   checkerboard_columns: " << COLUMNS << "\n";
    out << "   checkerboard_rows: " << ROWS << "\n";
    out << "   square_size_m: " << SQUARE_SIZE_M << "\n";
    out << "   original_sample_count: "
        << original_sample_count << "\n";
    out << "   accepted_sample_count: "
        << accepted_sample_count << "\n";
    out << "   rejected_sample_count: "
        << rejected_indices.size() << "\n";
    out << "   max_view_error_px: "
        << MAX_VIEW_ERROR_PX << "\n";
    out << "   rms: " << result.rms << "\n";
    out << "   mean_reprojection_error: "
        << result.mean_reprojection_error << "\n";

    out << "   accepted_original_sample_indices: [ ";
    for (std::size_t i = 0; i < accepted_indices.size(); ++i) {
        if (i)
            out << ", ";
        out << (accepted_indices[i] + 1);
    }
    out << " ]\n";

    out << "   rejected_original_sample_indices: [ ";
    for (std::size_t i = 0; i < rejected_indices.size(); ++i) {
        if (i)
            out << ", ";
        out << (rejected_indices[i] + 1);
    }
    out << " ]\n";

    out.flush();

    if (!out)
        throw std::runtime_error("Failed while writing output file: " + path);
}

bool robust_calibrate_and_save(
    const std::vector<std::vector<cv::Point2f>> &all_samples,
    const cv::Size image_size,
    const PoseTemplate &pose_template,
    const std::string &serial,
    const std::string &output_file)
{
    if (static_cast<int>(all_samples.size()) < MIN_SAMPLES) {
        std::cout << "\nNeed at least " << MIN_SAMPLES
                  << " samples; currently have "
                  << all_samples.size() << ".\n";
        return false;
    }

    std::vector<std::vector<cv::Point2f>> samples = all_samples;

    std::vector<int> original_indices(samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i)
        original_indices[i] = static_cast<int>(i);

    std::vector<int> rejected_original_indices;
    CalibrationResult result;

    for (int round = 0;
         round < MAX_REJECTION_ROUNDS;
         ++round)
    {
        result = calibrate_once(samples, image_size);

        int worst_idx = -1;
        double worst_error = -1.0;

        for (std::size_t i = 0;
             i < result.per_view_errors.size();
             ++i)
        {
            if (result.per_view_errors[i] > worst_error) {
                worst_error = result.per_view_errors[i];
                worst_idx = static_cast<int>(i);
            }
        }

        std::cout
            << "Round " << (round + 1)
            << ": n=" << samples.size()
            << " RMS=" << result.rms
            << " mean=" << result.mean_reprojection_error
            << " px"
            << " worst=" << worst_error
            << " px\n";

        if (worst_error <= MAX_VIEW_ERROR_PX)
            break;

        if (static_cast<int>(samples.size()) <= MIN_SAMPLES) {
            std::cout
                << "Reached minimum sample count; "
                << "stopping rejection.\n";
            break;
        }

        rejected_original_indices.push_back(
            original_indices[worst_idx]);

        samples.erase(
            samples.begin() + worst_idx);

        original_indices.erase(
            original_indices.begin() + worst_idx);
    }

    result = calibrate_once(samples, image_size);

    write_pose_config(
        output_file,
        result,
        pose_template,
        serial,
        image_size,
        static_cast<int>(all_samples.size()),
        static_cast<int>(samples.size()),
        original_indices,
        rejected_original_indices);

    std::cout << "\n===== FINAL CALIBRATION =====\n";
    std::cout << "Accepted: "
              << samples.size() << " / "
              << all_samples.size() << "\n";
    std::cout << "RMS: "
              << result.rms << " px\n";
    std::cout << "Mean reprojection error: "
              << result.mean_reprojection_error
              << " px\n";
    std::cout << "K:\n"
              << result.K << "\n";
    std::cout << "D:\n"
              << result.D << "\n";
    std::cout << "Tracker-ready config saved to: "
              << output_file << "\n";

    return true;
}

void detector_thread() {
    const cv::Size pattern(COLUMNS, ROWS);

    while (g_running.load(std::memory_order_relaxed)) {
        if (!g_detect_image_ready.load(
                std::memory_order_acquire))
        {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(1));
            continue;
        }

        cv::Mat full;

        {
            std::lock_guard<std::mutex> lock(
                g_detect_input_mutex);

            if (g_newest_detect_image.empty()) {
                g_detect_image_ready.store(
                    false,
                    std::memory_order_release);
                continue;
            }

            full = g_newest_detect_image.clone();

            g_detect_image_ready.store(
                false,
                std::memory_order_release);
        }

        cv::Mat denoised, binary, small;

        cv::GaussianBlur(
            full,
            denoised,
            cv::Size(5, 5),
            1.2);

        cv::resize(
            denoised,
            small,
            cv::Size(),
            DETECT_SCALE,
            DETECT_SCALE,
            cv::INTER_AREA);

        std::vector<cv::Point2f> corners_small;

        const auto t0 =
            std::chrono::steady_clock::now();

        bool found =
            cv::findChessboardCornersSB(
                small,
                pattern,
                corners_small,
                cv::CALIB_CB_NORMALIZE_IMAGE |
                cv::CALIB_CB_ACCURACY);

        if (!found) {
            corners_small.clear();

            cv::threshold(
                denoised,
                binary,
                0,
                255,
                cv::THRESH_BINARY |
                cv::THRESH_OTSU);

            const cv::Mat kernel =
                cv::getStructuringElement(
                    cv::MORPH_RECT,
                    cv::Size(3, 3));

            cv::morphologyEx(
                binary,
                binary,
                cv::MORPH_CLOSE,
                kernel);

            cv::resize(
                binary,
                small,
                cv::Size(),
                DETECT_SCALE,
                DETECT_SCALE,
                cv::INTER_AREA);

            found =
                cv::findChessboardCornersSB(
                    small,
                    pattern,
                    corners_small,
                    cv::CALIB_CB_NORMALIZE_IMAGE |
                    cv::CALIB_CB_EXHAUSTIVE |
                    cv::CALIB_CB_ACCURACY);
        }

        const auto t1 =
            std::chrono::steady_clock::now();

        const double detect_ms =
            std::chrono::duration<double, std::milli>(
                t1 - t0).count();

        std::vector<cv::Point2f> corners_full;

        if (found) {
            corners_full.reserve(
                corners_small.size());

            const float inv_scale =
                static_cast<float>(
                    1.0 / DETECT_SCALE);

            for (const auto &p : corners_small) {
                corners_full.emplace_back(
                    p.x * inv_scale,
                    p.y * inv_scale);
            }

            cv::cornerSubPix(
                denoised,
                corners_full,
                cv::Size(4, 4),
                cv::Size(-1, -1),
                cv::TermCriteria(
                    cv::TermCriteria::EPS |
                    cv::TermCriteria::COUNT,
                    30,
                    0.005));
        }

        {
            std::lock_guard<std::mutex> lock(
                g_result_mutex);

            g_found = found;
            g_latest_corners =
                std::move(corners_full);

            g_detect_ms = detect_ms;
            g_result_time =
                std::chrono::steady_clock::now();
        }
    }
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options options =
            parse_args(argc, argv);

        const PoseTemplate pose_template =
            load_pose_template(
                options.pose_template);

        std::cout << "Opening camera "
                  << options.serial << "...\n";

        Metavision::Camera cam =
            Metavision::Camera::from_serial(
                options.serial);

        const auto &geometry =
            cam.geometry();

        const int width =
            geometry.get_width();

        const int height =
            geometry.get_height();

        std::cout << "Resolution: "
                  << width << " x "
                  << height << "\n";

        std::cout << "Checkerboard: "
                  << COLUMNS << "x"
                  << ROWS
                  << " inner corners\n";

        std::cout
            << "\nCapture at least 30-35 diverse views:\n"
            << "  - center + all image corners/edges\n"
            << "  - several distances\n"
            << "  - strong left/right/up/down tilts\n"
            << "  - rotate board in-plane too\n\n"
            << "SPACE capture | C calibrate/save | "
            << "X clear | Q/Esc quit\n";

        g_events.reserve(500'000);

        cam.cd().add_callback(
            [&](const Metavision::EventCD *begin,
                const Metavision::EventCD *end)
            {
                const std::size_t n =
                    static_cast<std::size_t>(
                        end - begin);

                if (!n)
                    return;

                std::lock_guard<std::mutex> lock(
                    g_events_mutex);

                if (g_events.size() + n >
                    MAX_EVENTS)
                {
                    g_events.clear();
                }

                const std::size_t old =
                    g_events.size();

                g_events.resize(old + n);

                for (std::size_t i = 0;
                     i < n;
                     ++i)
                {
                    const auto &e = begin[i];

                    g_events[old + i] = {
                        static_cast<std::uint16_t>(e.x),
                        static_cast<std::uint16_t>(e.y),
                        static_cast<std::uint8_t>(e.p)
                    };
                }
            });

        cam.start();

        std::thread detector(detector_thread);

        cv::Mat live(
            height,
            width,
            CV_8UC1,
            cv::Scalar(0));

        cv::Mat display;

        cv::Mat detection_activity(
            height,
            width,
            CV_8UC1,
            cv::Scalar(0));

        std::vector<PackedEvent> local;
        local.reserve(500'000);

        auto last_clear =
            std::chrono::steady_clock::now();

        auto last_detect_submit =
            last_clear;

        cv::namedWindow(
            "Event checkerboard -> pose config",
            cv::WINDOW_NORMAL);

        cv::resizeWindow(
            "Event checkerboard -> pose config",
            width,
            height);

        std::vector<
            std::vector<cv::Point2f>>
            samples;

        samples.reserve(50);

        while (cam.is_running()) {
            {
                std::lock_guard<std::mutex> lock(
                    g_events_mutex);

                local.swap(g_events);
            }

            const auto now =
                std::chrono::steady_clock::now();

            if (now - last_clear >=
                std::chrono::milliseconds(
                    LIVE_PERSISTENCE_MS))
            {
                live.setTo(0);
                last_clear = now;
            }

            for (const auto &e : local) {
                live.at<std::uint8_t>(
                    e.y, e.x) =
                    e.p ? 255 : 100;

                std::uint8_t &activity =
                    detection_activity
                        .at<std::uint8_t>(
                            e.y, e.x);

                activity =
                    cv::saturate_cast<
                        std::uint8_t>(
                            static_cast<int>(
                                activity) +
                            DETECTION_EVENT_INCREMENT);
            }

            local.clear();

            if (now - last_detect_submit >=
                    std::chrono::milliseconds(
                        DETECTION_ACCUMULATION_MS)
                &&
                !g_detect_image_ready.load(
                    std::memory_order_acquire))
            {
                {
                    std::lock_guard<std::mutex> lock(
                        g_detect_input_mutex);

                    detection_activity.copyTo(
                        g_newest_detect_image);
                }

                detection_activity.setTo(0);

                g_detect_image_ready.store(
                    true,
                    std::memory_order_release);

                last_detect_submit = now;
            }

            cv::cvtColor(
                live,
                display,
                cv::COLOR_GRAY2BGR);

            bool found = false;
            double detect_ms = 0.0;
            double result_age_ms = 1e9;
            std::vector<cv::Point2f> corners;

            {
                std::lock_guard<std::mutex> lock(
                    g_result_mutex);

                found = g_found;
                detect_ms = g_detect_ms;
                corners = g_latest_corners;

                if (g_result_time
                        .time_since_epoch()
                        .count() != 0)
                {
                    result_age_ms =
                        std::chrono::duration<
                            double,
                            std::milli>(
                                now -
                                g_result_time)
                            .count();
                }
            }

            const bool result_fresh =
                result_age_ms < 500.0;

            if (found && result_fresh) {
                cv::drawChessboardCorners(
                    display,
                    cv::Size(COLUMNS, ROWS),
                    corners,
                    true);
            }

            cv::putText(
                display,
                (found && result_fresh)
                    ? "FOUND 9x6 - SPACE capture"
                    : "searching checkerboard",
                cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX,
                0.75,
                (found && result_fresh)
                    ? cv::Scalar(0, 255, 0)
                    : cv::Scalar(255, 255, 255),
                2,
                cv::LINE_AA);

            cv::putText(
                display,
                "detect " +
                    cv::format("%.1f", detect_ms) +
                    " ms",
                cv::Point(10, 58),
                cv::FONT_HERSHEY_SIMPLEX,
                0.55,
                cv::Scalar(200, 200, 200),
                1,
                cv::LINE_AA);

            cv::putText(
                display,
                "samples " +
                    std::to_string(samples.size()) +
                    " / " +
                    std::to_string(TARGET_SAMPLES) +
                    " | C calibrate/save | X clear",
                cv::Point(10, height - 18),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::Scalar(0, 255, 255),
                1,
                cv::LINE_AA);

            cv::imshow(
                "Event checkerboard -> pose config",
                display);

            const int key =
                cv::waitKey(1) & 0xFF;

            if (key == 'q' || key == 27)
                break;

            if (key == 'x' || key == 'X') {
                samples.clear();
                std::cout << "Samples cleared.\n";
            }

            if (key == 'c' || key == 'C') {
                robust_calibrate_and_save(
                    samples,
                    cv::Size(width, height),
                    pose_template,
                    options.serial,
                    options.output);
            }

            if (key == 32) {
                if (found &&
                    result_fresh &&
                    static_cast<int>(
                        corners.size()) ==
                        COLUMNS * ROWS)
                {
                    samples.push_back(corners);

                    std::cout
                        << "Captured sample "
                        << samples.size()
                        << "\n";
                }
                else {
                    std::cout
                        << "No fresh valid checkerboard result.\n";
                }
            }
        }

        g_running.store(
            false,
            std::memory_order_relaxed);

        cam.stop();

        if (detector.joinable())
            detector.join();

        cv::destroyAllWindows();

        return 0;
    }
    catch (const std::exception &e) {
        std::cerr << "Error: "
                  << e.what() << "\n";
        return 1;
    }
}
