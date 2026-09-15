#include <metavision/sdk/stream/camera.h>
#include <metavision/sdk/base/events/event_cd.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <chrono>
#include <vector>

// ============================================================
// FILTER PARAMETERS TO TUNE
// ============================================================

constexpr std::array<double, 3> LED_FREQ_HZ = {
    165.0,
    366.0,
    596.0
};

constexpr std::size_t HISTORY_SIZE = 4;
constexpr int MAX_SAME_POLARITY_MULTIPLE = 3;
constexpr int MAX_CROSS_POLARITY_HALF_MULTIPLE = 5;
constexpr double PERIOD_TOLERANCE = 0.03;
constexpr int SAME_POLARITY_WEIGHT = 2;
constexpr int CROSS_POLARITY_WEIGHT = 1;
constexpr int MIN_SCORE = 4;
constexpr bool PRINT_ACCEPTED_EVENTS = false;

// ============================================================
// DISPLAY DEFAULTS
// ============================================================

constexpr int DEFAULT_DISPLAY_FPS = 60;
constexpr int DEFAULT_POINT_RADIUS = 1;

struct RuntimeOptions {
    bool visualize = false;
    int display_fps = DEFAULT_DISPLAY_FPS;
    int point_radius = DEFAULT_POINT_RADIUS;
};

static RuntimeOptions parse_options(int argc, char **argv) {
    RuntimeOptions opt;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--visualize") == 0) {
            opt.visualize = true;
        } else if (std::strcmp(argv[i], "--display-fps") == 0 && i + 1 < argc) {
            opt.display_fps = std::max(1, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--point-radius") == 0 && i + 1 < argc) {
            opt.point_radius = std::max(0, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::cout
                << "Usage: live_frequency_ring [options]\n"
                << "  --visualize           Show accepted events in real time\n"
                << "  --display-fps N       Display refresh rate (default 60)\n"
                << "  --point-radius N      Accepted-event radius (default 1)\n";
            std::exit(0);
        }
    }

    return opt;
}

// ============================================================
// CIRCULAR TIMESTAMP BUFFER
// ============================================================

template <std::size_t N>
struct TimestampRing {
    std::array<uint32_t, N> ts{};
    uint8_t head = 0;
    uint8_t count = 0;

    inline void push(uint32_t t) {
        ts[head] = t;
        ++head;
        if (head == N) {
            head = 0;
        }
        if (count < N) {
            ++count;
        }
    }

    inline uint32_t get(std::size_t age) const {
        int index = static_cast<int>(head) - 1 - static_cast<int>(age);
        while (index < 0) {
            index += static_cast<int>(N);
        }
        return ts[static_cast<std::size_t>(index)];
    }
};

struct PixelState {
    TimestampRing<HISTORY_SIZE> pos;
    TimestampRing<HISTORY_SIZE> neg;
};

struct FilterResult {
    bool passed = false;
    int frequency_id = -1;
    int same_matches = 0;
    int cross_matches = 0;
    int score = 0;
    double mean_error = std::numeric_limits<double>::infinity();
};

inline double period_us(double frequency_hz) {
    return 1e6 / frequency_hz;
}

inline bool same_polarity_match(uint32_t dt,
                                double frequency_hz,
                                double &relative_error) {
    const double T = period_us(frequency_hz);
    bool matched = false;
    double best = std::numeric_limits<double>::infinity();

    for (int multiple = 1; multiple <= MAX_SAME_POLARITY_MULTIPLE; ++multiple) {
        const double expected = T * static_cast<double>(multiple);
        const double err = std::abs(static_cast<double>(dt) - expected) / expected;

        if (err <= PERIOD_TOLERANCE && err < best) {
            best = err;
            matched = true;
        }
    }

    relative_error = best;
    return matched;
}

inline bool cross_polarity_match(uint32_t dt,
                                 double frequency_hz,
                                 double &relative_error) {
    const double T = period_us(frequency_hz);
    bool matched = false;
    double best = std::numeric_limits<double>::infinity();

    for (int odd = 1; odd <= MAX_CROSS_POLARITY_HALF_MULTIPLE; odd += 2) {
        const double expected = 0.5 * T * static_cast<double>(odd);
        const double err = std::abs(static_cast<double>(dt) - expected) / expected;

        if (err <= PERIOD_TOLERANCE && err < best) {
            best = err;
            matched = true;
        }
    }

    relative_error = best;
    return matched;
}

template <std::size_t N>
inline FilterResult classify_event(const TimestampRing<N> &same_history,
                                   const TimestampRing<N> &opposite_history,
                                   uint32_t now) {
    FilterResult best;

    for (std::size_t f = 0; f < LED_FREQ_HZ.size(); ++f) {
        int same_matches = 0;
        int cross_matches = 0;
        double total_error = 0.0;
        int error_count = 0;

        for (std::size_t age = 0; age < same_history.count; ++age) {
            const uint32_t dt = now - same_history.get(age);
            double error = 0.0;

            if (same_polarity_match(dt, LED_FREQ_HZ[f], error)) {
                ++same_matches;
                total_error += error;
                ++error_count;
            }
        }

        for (std::size_t age = 0; age < opposite_history.count; ++age) {
            const uint32_t dt = now - opposite_history.get(age);
            double error = 0.0;

            if (cross_polarity_match(dt, LED_FREQ_HZ[f], error)) {
                ++cross_matches;
                total_error += error;
                ++error_count;
            }
        }

        const int score = SAME_POLARITY_WEIGHT * same_matches
                        + CROSS_POLARITY_WEIGHT * cross_matches;

        if (score < MIN_SCORE || error_count == 0) {
            continue;
        }

        const double mean_error = total_error / static_cast<double>(error_count);

        if (!best.passed || score > best.score ||
            (score == best.score && mean_error < best.mean_error)) {
            best.passed = true;
            best.frequency_id = static_cast<int>(f);
            best.same_matches = same_matches;
            best.cross_matches = cross_matches;
            best.score = score;
            best.mean_error = mean_error;
        }
    }

    return best;
}

// ============================================================
// MAIN
// ============================================================

int main(int argc, char **argv) {
    using namespace Metavision;

    const RuntimeOptions options = parse_options(argc, argv);

    Camera camera;

    try {
        camera = Camera::from_first_available();
    } catch (const std::exception &e) {
        std::cerr << "Failed to open Prophesee EVK4: " << e.what() << '\n';
        return 1;
    }

    const auto &geometry = camera.geometry();
    const int width = geometry.get_width();
    const int height = geometry.get_height();

    std::cout << "Camera: " << width << " x " << height << '\n';
    std::cout << "History per polarity: " << HISTORY_SIZE << '\n';
    std::cout << "Tolerance: +/-" << PERIOD_TOLERANCE * 100.0 << "%\n";
    std::cout << "Minimum score: " << MIN_SCORE << '\n';
    std::cout << "Visualization: " << (options.visualize ? "ON" : "OFF") << '\n';
    if (options.visualize) {
        std::cout << "Display FPS: " << options.display_fps << '\n';
    }

    std::cout << "\nTarget frequencies:\n";
    for (double f : LED_FREQ_HZ) {
        std::cout << "  " << f << " Hz -> T = " << period_us(f) << " us\n";
    }

    std::vector<PixelState> pixels(
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height));

    std::atomic<uint64_t> total_events{0};
    std::atomic<uint64_t> passed_events{0};
    std::array<std::atomic<uint64_t>, LED_FREQ_HZ.size()> frequency_counts{};

    // Visualization buffers. The camera callback never calls OpenCV GUI
    // functions. It only publishes the newest accumulation to ready_frame.
    // The main thread owns namedWindow/imshow/waitKey.
    cv::Mat accumulation;
    cv::Mat ready_frame;
    std::mutex display_mutex;
    std::atomic<bool> frame_ready{false};
    uint32_t last_publish_ts = 0;

    const uint32_t display_period_us =
        static_cast<uint32_t>(1'000'000 / options.display_fps);

    if (options.visualize) {
        accumulation = cv::Mat::zeros(height, width, CV_8UC3);
        ready_frame = cv::Mat::zeros(height, width, CV_8UC3);
    }

    std::atomic<bool> stop_requested{false};

    camera.cd().add_callback(
        [&](const EventCD *begin, const EventCD *end) {
            for (const EventCD *ev = begin; ev != end; ++ev) {
                ++total_events;

                const std::size_t index =
                    static_cast<std::size_t>(ev->y) * static_cast<std::size_t>(width) +
                    static_cast<std::size_t>(ev->x);

                PixelState &pixel = pixels[index];

                auto &same_history = ev->p ? pixel.pos : pixel.neg;
                auto &opposite_history = ev->p ? pixel.neg : pixel.pos;

                const uint32_t timestamp = static_cast<uint32_t>(ev->t);

                const FilterResult result =
                    classify_event(same_history, opposite_history, timestamp);

                // Keep all raw events in their polarity-specific rings.
                same_history.push(timestamp);

                if (result.passed) {
                    ++passed_events;
                    ++frequency_counts[static_cast<std::size_t>(result.frequency_id)];

                    if (options.visualize) {
                        // BGR colors:
                        // 165 Hz = green, 366 Hz = yellow, 596 Hz = blue.
                        static const std::array<cv::Vec3b, 3> COLORS = {
                            cv::Vec3b(0, 255, 0),
                            cv::Vec3b(0, 255, 255),
                            cv::Vec3b(255, 0, 0)
                        };

                        if (options.point_radius <= 0) {
                            accumulation.at<cv::Vec3b>(ev->y, ev->x) =
                                COLORS[static_cast<std::size_t>(result.frequency_id)];
                        } else {
                            cv::circle(
                                accumulation,
                                cv::Point(ev->x, ev->y),
                                options.point_radius,
                                cv::Scalar(
                                    COLORS[static_cast<std::size_t>(result.frequency_id)][0],
                                    COLORS[static_cast<std::size_t>(result.frequency_id)][1],
                                    COLORS[static_cast<std::size_t>(result.frequency_id)][2]),
                                cv::FILLED,
                                cv::LINE_8);
                        }
                    }

                    if constexpr (PRINT_ACCEPTED_EVENTS) {
                        std::cout << "PASS"
                                  << " x=" << ev->x
                                  << " y=" << ev->y
                                  << " p=" << (ev->p ? '+' : '-')
                                  << " t=" << ev->t
                                  << " f=" << LED_FREQ_HZ[result.frequency_id]
                                  << "Hz"
                                  << " score=" << result.score
                                  << " error=" << result.mean_error
                                  << '\n';
                    }
                }

                // Publish at most one newest display frame at the requested
                // camera-time cadence. There is no frame queue: ready_frame
                // is overwritten with the newest state if the GUI is slower.
                if (options.visualize &&
                    (last_publish_ts == 0 ||
                     timestamp - last_publish_ts >= display_period_us)) {
                    {
                        std::lock_guard<std::mutex> lock(display_mutex);
                        accumulation.copyTo(ready_frame);
                        frame_ready.store(true, std::memory_order_release);
                    }

                    accumulation.setTo(cv::Scalar(0, 0, 0));
                    last_publish_ts = timestamp;
                }
            }
        });

    camera.start();

    std::cout << "\nLive frequency filtering running.\n";
    if (options.visualize) {
        std::cout << "Press Q or Esc in the window to stop.\n";

        cv::namedWindow("Frequency-filtered events", cv::WINDOW_NORMAL);
        cv::Mat gui_frame = cv::Mat::zeros(height, width, CV_8UC3);

        while (!stop_requested.load()) {
            if (frame_ready.exchange(false, std::memory_order_acq_rel)) {
                {
                    std::lock_guard<std::mutex> lock(display_mutex);
                    ready_frame.copyTo(gui_frame);
                }
                cv::imshow("Frequency-filtered events", gui_frame);
            }

            const int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {
                stop_requested.store(true);
                break;
            }

            // Avoid spinning at 100% CPU when there is no new frame.
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } else {
        std::cout << "Press ENTER to stop.\n";
        std::cin.get();
    }

    camera.stop();

    if (options.visualize) {
        cv::destroyAllWindows();
    }

    const uint64_t total = total_events.load();
    const uint64_t passed = passed_events.load();

    std::cout << "\n============================\n";
    std::cout << "RESULTS\n";
    std::cout << "============================\n";
    std::cout << "Total events  : " << total << '\n';
    std::cout << "Passed events : " << passed << '\n';

    if (total > 0) {
        std::cout << "Pass ratio    : "
                  << (100.0 * static_cast<double>(passed) /
                      static_cast<double>(total))
                  << "%\n";
    }

    std::cout << "\nAccepted by frequency:\n";
    for (std::size_t i = 0; i < LED_FREQ_HZ.size(); ++i) {
        std::cout << "  " << LED_FREQ_HZ[i] << " Hz : "
                  << frequency_counts[i].load() << '\n';
    }

    return 0;
}
