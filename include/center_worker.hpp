#pragma once

#include "spsc_ring.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>

namespace event_led_pose {

constexpr std::size_t CENTER_FREQ_COUNT = 3;

struct CenterSample {
    std::uint16_t x = 0;
    std::uint16_t y = 0;
    std::uint32_t t = 0;
    std::uint8_t frequency_id = 0;
    std::uint8_t pad[3]{};
};
static_assert(sizeof(CenterSample) == 12);

struct FrequencyCenter {
    bool valid = false;
    float x = 0.0f;
    float y = 0.0f;
    std::uint32_t sample_count = 0;
    std::uint32_t newest_t = 0;
};

struct CenterSnapshot {
    std::array<FrequencyCenter, CENTER_FREQ_COUNT> frequency{};
    std::uint32_t window_us = 0;
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
        // Event-time window used to estimate the current LED center.
        std::uint32_t window_us = 20000;

        // Publish center at most this often.
        std::uint32_t update_period_us = 5000; // 200 Hz

        // Hard memory/CPU bound for huge close-up LEDs.
        std::size_t max_samples_per_frequency = 20000;

        // Lossy worker-input transport.
        std::size_t input_ring_capacity = 1u << 17;

        // Do not display a center from only a few accidental detections.
        std::uint32_t min_samples = 8;

        // If absolutely no new classified samples arrive, hide stale centers.
        std::uint32_t wall_stale_ms = 100;
    };

    CenterWorker(const Config &config, CenterStore &store);
    ~CenterWorker();

    void start();
    void stop();

    // Single producer: frequency worker.
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

private:
    struct FrequencyWindow {
        std::deque<CenterSample> samples;
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

    std::array<FrequencyWindow, CENTER_FREQ_COUNT> windows_{};

    void run();
    bool consume_available(std::uint32_t &global_newest_t);
    void trim_windows(std::uint32_t global_newest_t);
    void publish_snapshot();
};

} // namespace event_led_pose
