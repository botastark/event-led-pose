#pragma once

#include "spsc_ring.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace event_led_pose {

constexpr std::size_t CENTER_FREQ_COUNT = 3;
constexpr std::size_t RADIAL_HISTOGRAM_BINS = 64;

struct CenterSample {
    std::uint16_t x = 0;
    std::uint16_t y = 0;
    std::uint32_t t = 0;
    std::uint8_t frequency_id = 0;
    std::uint8_t pad[3]{};
};
static_assert(sizeof(CenterSample) == 12);

// One live spatial snapshot for one known LED frequency.
struct FrequencyCenter {
    bool valid = false;

    float x = 0.0f;
    float y = 0.0f;

    // Radial distribution around (x,y), in pixels.
    float mean_radius = 0.0f;
    float std_radius = 0.0f;
    float rms_radius = 0.0f;
    float p95_radius = 0.0f;
    float max_radius = 0.0f;

    // Raw event counts by radial-distance bin.
    // Unlike the previous normalized distribution, these remain counts
    // so the three live plots can share one meaningful Y axis.
    std::array<std::uint32_t, RADIAL_HISTOGRAM_BINS> radial_histogram{};

    std::uint32_t sample_count = 0;
    std::uint32_t newest_t = 0;
};

struct CenterSnapshot {
    std::array<FrequencyCenter, CENTER_FREQ_COUNT> frequency{};

    std::uint32_t window_us = 0;
    std::uint32_t newest_t = 0;

    // Shared X-axis range used by all three histogram panels.
    float histogram_max_radius_px = 400.0f;
};

class CenterStore {
public:
    void publish(const CenterSnapshot &snapshot);

    bool copy_if_new(std::uint64_t &last_version,
                     CenterSnapshot &out) const;

private:
    mutable std::mutex mutex_;
    CenterSnapshot snapshot_;
    std::atomic<std::uint64_t> version_{0};
};

class CenterWorker {
public:
    struct Config {
        // Sliding event-camera-time window.
        // For motion experiments start around 5-10 ms.
        std::uint32_t window_us = 10000;

        // Spatial stats publication period.
        // 100 Hz by default.
        std::uint32_t update_period_us = 10000;

        // Hard memory/CPU bound for very large close-up LEDs.
        std::size_t max_samples_per_frequency = 20000;

        // Frequency worker -> stats worker SPSC queue.
        std::size_t input_ring_capacity = 1u << 17;

        // Reject tiny accidental clusters.
        std::uint32_t min_samples = 8;

        // Radial histogram X-axis: [0, histogram_max_radius_px].
        // The last bin also collects radii beyond this bound.
        float histogram_max_radius_px = 400.0f;

        // Hide a center if no fresh classified samples arrive.
        std::uint32_t wall_stale_ms = 100;

        // Optional CSV recording.
        // Empty path = disabled.
        std::string csv_path;

        // Flush CSV occasionally rather than every row.
        std::uint32_t csv_flush_period_ms = 1000;
    };

    CenterWorker(const Config &config, CenterStore &store);
    ~CenterWorker();

    void start();
    void stop();

    // Single producer: frequency worker.
    // These calls do no statistics or file I/O.
    void begin_batch() noexcept;

    bool push(std::uint16_t x,
              std::uint16_t y,
              std::uint32_t t,
              std::uint8_t frequency_id) noexcept;

    void end_batch() noexcept;

    std::uint64_t submitted() const noexcept {
        return submitted_.load(std::memory_order_relaxed);
    }

    std::uint64_t input_drops() const noexcept {
        return input_drops_.load(std::memory_order_relaxed);
    }

    std::uint64_t window_drops() const noexcept {
        return window_drops_.load(std::memory_order_relaxed);
    }

    std::uint64_t snapshots_published() const noexcept {
        return snapshots_published_.load(std::memory_order_relaxed);
    }

    std::uint64_t csv_rows_written() const noexcept {
        return csv_rows_written_.load(std::memory_order_relaxed);
    }

private:
    struct FrequencyWindow {
        std::deque<CenterSample> samples;

        // Running center sums; cheap to update as window slides.
        std::uint64_t sum_x = 0;
        std::uint64_t sum_y = 0;

        std::uint32_t newest_t = 0;

        std::chrono::steady_clock::time_point last_wall_sample{};
    };

    Config config_;
    CenterStore &store_;

    SpscRing<CenterSample> input_;

    std::atomic<bool> stop_requested_{false};
    std::thread thread_;

    std::uint64_t batch_submitted_ = 0;
    std::uint64_t batch_dropped_ = 0;

    std::atomic<std::uint64_t> submitted_{0};
    std::atomic<std::uint64_t> input_drops_{0};
    std::atomic<std::uint64_t> window_drops_{0};
    std::atomic<std::uint64_t> snapshots_published_{0};
    std::atomic<std::uint64_t> csv_rows_written_{0};

    std::array<FrequencyWindow, CENTER_FREQ_COUNT> windows_{};

    // Reused scratch memory: no per-snapshot allocation.
    std::vector<float> radial_scratch_;

    std::ofstream csv_;
    std::chrono::steady_clock::time_point last_csv_flush_{};

    void run();

    bool consume_available(std::uint32_t &global_newest_t);

    void trim_windows(std::uint32_t global_newest_t);

    FrequencyCenter compute_frequency_stats(
        const FrequencyWindow &window);

    void publish_snapshot(std::uint32_t global_newest_t);

    void open_csv();
    void write_csv_snapshot(const CenterSnapshot &snapshot);
    void maybe_flush_csv();
};

} // namespace event_led_pose
