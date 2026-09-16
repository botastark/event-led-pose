#include "fast_event_viewer.hpp"

#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/stream/camera.h>

#include <atomic>
#include <cstdint>
#include <iostream>

int main() {
    using namespace Metavision;
    using event_led_pose::FastEventViewer;

    Camera camera = Camera::from_first_available();

    const auto &geometry = camera.geometry();
    const int width = geometry.get_width();
    const int height = geometry.get_height();

    FastEventViewer::Config cfg;
    cfg.sensor_width = width;
    cfg.sensor_height = height;
    cfg.title = "EVK4 direct event stream";
    cfg.vsync = false;
    cfg.max_fps = 0;                  // render as soon as possible
    cfg.point_size = 1.0f;
    cfg.max_points_per_frame = 200000;

    FastEventViewer viewer(cfg);

    std::atomic<std::uint64_t> total_events{0};

    camera.cd().add_callback(
        [&](const EventCD *begin, const EventCD *end) {
            total_events.fetch_add(
                static_cast<std::uint64_t>(end - begin),
                std::memory_order_relaxed);

            // This demo deliberately pushes raw events.
            // Positive = white, negative = dim gray.
            //
            // For a detector/filter, call viewer.push() only for whatever
            // functionality you want to inspect.
            for (const EventCD *ev = begin; ev != end; ++ev) {
                if (ev->p) {
                    viewer.push(ev->x, ev->y, 255, 255, 255);
                } else {
                    viewer.push(ev->x, ev->y, 90, 90, 90);
                }
            }
        });

    camera.start();

    // GLFW/OpenGL loop stays on the main thread.
    viewer.run();

    camera.stop();

    const auto stats = viewer.stats();

    std::cout
        << "\nTotal camera events : " << total_events.load() << '\n'
        << "Viewer submitted    : " << stats.submitted << '\n'
        << "Ring-full drops     : " << stats.ring_full_drops << '\n'
        << "Stale display drops : " << stats.stale_drops << '\n'
        << "Points rendered     : " << stats.rendered << '\n'
        << "Display frames      : " << stats.frames << '\n';

    return 0;
}
