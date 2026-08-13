#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/stream/camera.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

volatile std::sig_atomic_t stop_requested = 0;

void signal_handler(int) {
    stop_requested = 1;
}

struct Options {
    double duration_s = 30.0;
    std::optional<std::filesystem::path> input;
    std::optional<std::filesystem::path> record;
    std::optional<std::filesystem::path> settings;
    bool overwrite = false;
};

void print_usage(const char *program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " [options]\n\n"
        << "Options:\n"
        << "  --duration-s SECONDS   Wall-clock duration; 0 means until Ctrl-C\n"
        << "  --input FILE.raw       Replay a recording instead of using EVK4\n"
        << "  --record FILE.raw      Record the live stream\n"
        << "  --settings FILE.bias   Load Metavision camera settings\n"
        << "  --overwrite            Permit replacing an existing recording\n"
        << "  --help                 Show this message\n";
}

std::string require_value(int &index, int argc, char **argv) {
    if (index + 1 >= argc) {
        throw std::runtime_error(
            "Missing value after " + std::string(argv[index]));
    }
    return argv[++index];
}

Options parse_options(int argc, char **argv) {
    Options options;

    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];

        if (argument == "--duration-s") {
            options.duration_s = std::stod(require_value(i, argc, argv));
        } else if (argument == "--input") {
            options.input = require_value(i, argc, argv);
        } else if (argument == "--record") {
            options.record = require_value(i, argc, argv);
        } else if (argument == "--settings") {
            options.settings = require_value(i, argc, argv);
        } else if (argument == "--overwrite") {
            options.overwrite = true;
        } else if (argument == "--help") {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::runtime_error("Unknown option: " + argument);
        }
    }

    if (options.duration_s < 0.0) {
        throw std::runtime_error("--duration-s cannot be negative");
    }

    if (options.input && options.record) {
        throw std::runtime_error(
            "--record is intended for a live camera, not file replay");
    }

    return options;
}

struct Counters {
    std::atomic<std::uint64_t> total{0};
    std::atomic<std::uint64_t> on{0};
    std::atomic<std::uint64_t> off{0};
    std::atomic<std::uint64_t> callbacks{0};
    std::atomic<std::uint64_t> timestamp_regressions{0};
    std::atomic<std::uint64_t> maximum_batch{0};

    std::atomic<std::int64_t> first_timestamp_us{-1};
    std::atomic<std::int64_t> last_timestamp_us{-1};
};

void update_maximum(
    std::atomic<std::uint64_t> &destination,
    std::uint64_t value) {

    std::uint64_t current = destination.load(std::memory_order_relaxed);

    while (current < value &&
           !destination.compare_exchange_weak(
               current,
               value,
               std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse_options(argc, argv);

        if (options.input && !std::filesystem::exists(*options.input)) {
            throw std::runtime_error(
                "Input recording does not exist: " +
                options.input->string());
        }

        if (options.settings &&
            !std::filesystem::exists(*options.settings)) {
            throw std::runtime_error(
                "Settings file does not exist: " +
                options.settings->string());
        }

        if (options.record &&
            std::filesystem::exists(*options.record) &&
            !options.overwrite) {
            throw std::runtime_error(
                "Recording already exists: " +
                options.record->string() +
                "\nUse --overwrite only if replacement is intended.");
        }

        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        Metavision::Camera camera;

        if (options.input) {
            camera = Metavision::Camera::from_file(*options.input);
        } else {
            camera = Metavision::Camera::from_first_available();
        }

        if (options.settings && !camera.load(*options.settings)) {
            throw std::runtime_error(
                "Could not load camera settings: " +
                options.settings->string());
        }

        const int width = camera.geometry().get_width();
        const int height = camera.geometry().get_height();

        std::cout
            << "Input: "
            << (options.input ? options.input->string()
                              : std::string("first available live camera"))
            << '\n'
            << "Sensor geometry: " << width << " x " << height << '\n';

        Counters counters;

        // The SDK invokes this callback with timestamp-ordered buffers.
        // Every event is still inspected independently in the inner loop.
        Metavision::timestamp previous_timestamp_us = -1;

        camera.cd().add_callback(
            [&](const Metavision::EventCD *begin,
                const Metavision::EventCD *end) {

                if (begin == end) {
                    return;
                }

                std::uint64_t local_on = 0;
                std::uint64_t local_off = 0;
                std::uint64_t local_regressions = 0;

                for (const Metavision::EventCD *event = begin;
                     event != end;
                     ++event) {

                    if (previous_timestamp_us >= 0 &&
                        event->t < previous_timestamp_us) {
                        ++local_regressions;
                    }

                    previous_timestamp_us = event->t;

                    if (event->p != 0) {
                        ++local_on;
                    } else {
                        ++local_off;
                    }
                }

                const auto batch_size =
                    static_cast<std::uint64_t>(end - begin);

                std::int64_t expected_first = -1;
                counters.first_timestamp_us.compare_exchange_strong(
                    expected_first,
                    static_cast<std::int64_t>(begin->t),
                    std::memory_order_relaxed);

                counters.last_timestamp_us.store(
                    static_cast<std::int64_t>((end - 1)->t),
                    std::memory_order_relaxed);

                counters.total.fetch_add(
                    batch_size, std::memory_order_relaxed);
                counters.on.fetch_add(
                    local_on, std::memory_order_relaxed);
                counters.off.fetch_add(
                    local_off, std::memory_order_relaxed);
                counters.callbacks.fetch_add(
                    1, std::memory_order_relaxed);
                counters.timestamp_regressions.fetch_add(
                    local_regressions, std::memory_order_relaxed);

                update_maximum(counters.maximum_batch, batch_size);
            });

        if (!camera.start()) {
            throw std::runtime_error("Camera stream did not start");
        }

        bool recording_active = false;

        if (options.record) {
            if (!camera.start_recording(*options.record)) {
                camera.stop();
                throw std::runtime_error(
                    "Could not start RAW recording: " +
                    options.record->string());
            }

            recording_active = true;
            std::cout
                << "Recording: " << options.record->string() << '\n';
        }

        using Clock = std::chrono::steady_clock;

        const auto wall_start = Clock::now();
        auto previous_report_time = wall_start;
        auto next_report_time = wall_start + std::chrono::seconds(1);

        std::uint64_t previous_total = 0;

        std::cout << std::fixed << std::setprecision(3);

        while (!stop_requested && camera.is_running()) {
            const auto now = Clock::now();

            const double elapsed_s =
                std::chrono::duration<double>(now - wall_start).count();

            if (options.duration_s > 0.0 &&
                elapsed_s >= options.duration_s) {
                break;
            }

            if (now >= next_report_time) {
                const std::uint64_t total =
                    counters.total.load(std::memory_order_relaxed);

                const double interval_s =
                    std::chrono::duration<double>(
                        now - previous_report_time).count();

                const double event_rate_meps =
                    static_cast<double>(total - previous_total) /
                    interval_s / 1.0e6;

                const std::int64_t first_t =
                    counters.first_timestamp_us.load(
                        std::memory_order_relaxed);
                const std::int64_t last_t =
                    counters.last_timestamp_us.load(
                        std::memory_order_relaxed);

                const double sensor_elapsed_ms =
                    first_t >= 0 && last_t >= first_t
                        ? static_cast<double>(last_t - first_t) / 1000.0
                        : 0.0;

                std::cout
                    << "wall=" << elapsed_s << " s"
                    << "  sensor=" << sensor_elapsed_ms << " ms"
                    << "  events=" << total
                    << "  rate=" << event_rate_meps << " Mevent/s"
                    << "  ON=" << counters.on.load(
                           std::memory_order_relaxed)
                    << "  OFF=" << counters.off.load(
                           std::memory_order_relaxed)
                    << '\n';

                previous_total = total;
                previous_report_time = now;

                do {
                    next_report_time += std::chrono::seconds(1);
                } while (next_report_time <= now);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        if (recording_active) {
            camera.stop_recording(*options.record);
        }

        camera.stop();

        const double wall_duration_s =
            std::chrono::duration<double>(
                Clock::now() - wall_start).count();

        const std::uint64_t total =
            counters.total.load(std::memory_order_relaxed);
        const std::uint64_t on =
            counters.on.load(std::memory_order_relaxed);
        const std::uint64_t off =
            counters.off.load(std::memory_order_relaxed);

        const std::int64_t first_t =
            counters.first_timestamp_us.load(std::memory_order_relaxed);
        const std::int64_t last_t =
            counters.last_timestamp_us.load(std::memory_order_relaxed);

        const double sensor_duration_s =
            first_t >= 0 && last_t > first_t
                ? static_cast<double>(last_t - first_t) / 1.0e6
                : 0.0;

        std::cout
            << "\nFinal stream statistics\n"
            << "  wall duration: " << wall_duration_s << " s\n"
            << "  sensor duration: " << sensor_duration_s << " s\n"
            << "  total events: " << total << '\n'
            << "  ON events: " << on << '\n'
            << "  OFF events: " << off << '\n'
            << "  ON fraction: "
            << (total != 0 ? static_cast<double>(on) / total : 0.0)
            << '\n'
            << "  mean sensor rate: "
            << (sensor_duration_s > 0.0
                    ? static_cast<double>(total) /
                          sensor_duration_s / 1.0e6
                    : 0.0)
            << " Mevent/s\n"
            << "  callbacks: "
            << counters.callbacks.load(std::memory_order_relaxed)
            << '\n'
            << "  maximum callback batch: "
            << counters.maximum_batch.load(std::memory_order_relaxed)
            << '\n'
            << "  timestamp regressions: "
            << counters.timestamp_regressions.load(
                   std::memory_order_relaxed)
            << '\n';

        return counters.timestamp_regressions.load(
                   std::memory_order_relaxed) == 0
                   ? EXIT_SUCCESS
                   : EXIT_FAILURE;

    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}