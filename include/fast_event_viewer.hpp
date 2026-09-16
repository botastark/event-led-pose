#pragma once

#include "center_worker.hpp"
#include "packed_event.hpp"
#include "spsc_ring.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct GLFWwindow;

namespace event_led_pose {

class FastEventViewer {
public:
    struct Config {
        int sensor_width = 1280;
        int sensor_height = 720;

        std::size_t raw_ring_capacity  = 1u << 20;
        std::size_t freq_ring_capacity = 1u << 18;

        std::size_t max_raw_points_per_render  = 300000;
        std::size_t max_freq_points_per_render = 100000;

        std::uint32_t persistence_us = 6000;

        unsigned max_fps = 120;
        bool vsync = false;
        float point_size = 1.0f;

        // Colored + marker drawn at current frequency center.
        float center_marker_half_size_px = 10.0f;
        float center_marker_line_width = 2.0f;

        std::string title = "Raw + frequency + centers";
    };

    struct Stats {
        std::uint64_t raw_submitted = 0;
        std::uint64_t raw_ring_drops = 0;
        std::uint64_t raw_stale_drops = 0;

        std::uint64_t freq_submitted = 0;
        std::uint64_t freq_ring_drops = 0;
        std::uint64_t freq_stale_drops = 0;

        std::uint64_t rendered_frames = 0;
    };

    FastEventViewer(const Config &cfg,
                    CenterStore *center_store);

    ~FastEventViewer();

    void raw_begin_batch() noexcept;
    bool raw_push(const PackedEvent &e) noexcept;
    void raw_end_batch() noexcept;

    void freq_begin_batch() noexcept;
    bool freq_push(std::uint16_t x,
                   std::uint16_t y,
                   std::uint32_t t,
                   std::uint8_t frequency_id) noexcept;
    void freq_end_batch() noexcept;

    void run();

    void request_stop() noexcept {
        stop_requested_.store(true, std::memory_order_release);
    }

    Stats stats() const noexcept;

private:
    struct Point {
        std::uint16_t x;
        std::uint16_t y;
        std::uint32_t t;
        std::uint8_t r, g, b, pad;
    };
    static_assert(sizeof(Point) == 12);

    Config cfg_;
    CenterStore *center_store_ = nullptr;

    SpscRing<PackedEvent> raw_ring_;
    SpscRing<Point> freq_ring_;

    std::uint64_t raw_batch_submitted_ = 0;
    std::uint64_t raw_batch_dropped_ = 0;
    std::uint64_t freq_batch_submitted_ = 0;
    std::uint64_t freq_batch_dropped_ = 0;

    std::atomic<std::uint64_t> raw_submitted_{0};
    std::atomic<std::uint64_t> raw_ring_drops_{0};
    std::atomic<std::uint64_t> raw_stale_drops_{0};

    std::atomic<std::uint64_t> freq_submitted_{0};
    std::atomic<std::uint64_t> freq_ring_drops_{0};
    std::atomic<std::uint64_t> freq_stale_drops_{0};

    std::atomic<std::uint64_t> rendered_frames_{0};
    std::atomic<bool> stop_requested_{false};

    GLFWwindow *window_ = nullptr;

    unsigned int event_program_ = 0;
    unsigned int screen_program_ = 0;

    unsigned int event_vao_ = 0;
    unsigned int event_vbo_ = 0;

    unsigned int quad_vao_ = 0;
    unsigned int quad_vbo_ = 0;

    unsigned int canvas_texture_ = 0;
    unsigned int canvas_fbo_ = 0;

    std::vector<Point> raw_points_;
    std::vector<Point> freq_points_;
    std::vector<Point> center_lines_;

    bool canvas_initialized_ = false;
    std::uint32_t last_clear_ts_ = 0;

    CenterSnapshot center_snapshot_;
    std::uint64_t center_version_ = 0;

    void init_gl();
    void destroy_gl() noexcept;

    bool drain_raw();
    bool drain_freq();

    void draw_points(const std::vector<Point> &points,
                     unsigned int primitive);

    void draw_centers();

    void update_window_title();

    bool render_once();
};

} // namespace event_led_pose
