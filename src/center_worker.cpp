#include "center_worker.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <stdexcept>

namespace event_led_pose {

void CenterStore::publish(const CenterSnapshot &snapshot) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshot_ = snapshot;
    }

    version_.fetch_add(
        1,
        std::memory_order_release);
}

bool CenterStore::copy_if_new(
    std::uint64_t &last_version,
    CenterSnapshot &out) const
{
    const std::uint64_t version =
        version_.load(
            std::memory_order_acquire);

    if (version == last_version)
        return false;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        out = snapshot_;
    }

    last_version = version;
    return true;
}


CenterWorker::CenterWorker(
    const Config &config,
    CenterStore &store)
    : config_(config),
      store_(store),
      input_(config.input_ring_capacity)
{
    radial_scratch_.reserve(
        config_.max_samples_per_frequency);
}


CenterWorker::~CenterWorker() {
    stop();
}


void CenterWorker::start() {
    if (thread_.joinable())
        return;

    stop_requested_.store(
        false,
        std::memory_order_release);

    open_csv();

    thread_ =
        std::thread(
            &CenterWorker::run,
            this);
}


void CenterWorker::stop() {
    stop_requested_.store(
        true,
        std::memory_order_release);

    if (thread_.joinable())
        thread_.join();

    if (csv_.is_open()) {
        csv_.flush();
        csv_.close();
    }
}


void CenterWorker::begin_batch() noexcept {
    batch_submitted_ = 0;
    batch_dropped_ = 0;

    input_.producer_begin();
}


bool CenterWorker::push(
    std::uint16_t x,
    std::uint16_t y,
    std::uint32_t t,
    std::uint8_t frequency_id) noexcept
{
    ++batch_submitted_;

    if (frequency_id >= CENTER_FREQ_COUNT) {
        ++batch_dropped_;
        return false;
    }

    CenterSample sample;
    sample.x = x;
    sample.y = y;
    sample.t = t;
    sample.frequency_id = frequency_id;

    if (!input_.producer_push(sample)) {
        ++batch_dropped_;
        return false;
    }

    return true;
}


void CenterWorker::end_batch() noexcept {
    input_.producer_end();

    submitted_.fetch_add(
        batch_submitted_,
        std::memory_order_relaxed);

    input_drops_.fetch_add(
        batch_dropped_,
        std::memory_order_relaxed);
}


bool CenterWorker::consume_available(
    std::uint32_t &global_newest_t)
{
    std::uint64_t tail =
        input_.consumer_tail();

    const std::uint64_t head =
        input_.consumer_head();

    if (head == tail)
        return false;

    const auto wall_now =
        std::chrono::steady_clock::now();

    for (std::uint64_t seq = tail;
         seq < head;
         ++seq)
    {
        const CenterSample &sample =
            input_.consumer_at(seq);

        global_newest_t = sample.t;

        auto &window =
            windows_[sample.frequency_id];

        window.samples.push_back(sample);

        window.sum_x += sample.x;
        window.sum_y += sample.y;

        window.newest_t =
            sample.t;

        window.last_wall_sample =
            wall_now;

        // Hard cap: newest samples win.
        while (
            window.samples.size() >
            config_.max_samples_per_frequency)
        {
            const CenterSample old =
                window.samples.front();

            window.samples.pop_front();

            window.sum_x -= old.x;
            window.sum_y -= old.y;

            window_drops_.fetch_add(
                1,
                std::memory_order_relaxed);
        }
    }

    input_.consumer_commit(head);

    return true;
}


void CenterWorker::trim_windows(
    std::uint32_t global_newest_t)
{
    for (auto &window : windows_) {
        while (!window.samples.empty()) {
            const std::uint32_t age =
                global_newest_t -
                window.samples.front().t;

            if (age <= config_.window_us)
                break;

            const CenterSample old =
                window.samples.front();

            window.samples.pop_front();

            window.sum_x -= old.x;
            window.sum_y -= old.y;

            window_drops_.fetch_add(
                1,
                std::memory_order_relaxed);
        }
    }
}


FrequencyCenter
CenterWorker::compute_frequency_stats(
    const FrequencyWindow &window)
{
    FrequencyCenter result;

    const std::size_t n =
        window.samples.size();

    if (n < config_.min_samples)
        return result;

    if (
        window.last_wall_sample
            .time_since_epoch()
            .count()
        == 0)
    {
        return result;
    }

    const auto now =
        std::chrono::steady_clock::now();

    const auto stale_ms =
        std::chrono::duration_cast<
            std::chrono::milliseconds>(
            now -
            window.last_wall_sample)
            .count();

    if (
        stale_ms >
        static_cast<long long>(
            config_.wall_stale_ms))
    {
        return result;
    }

    const double cx =
        static_cast<double>(
            window.sum_x)
        /
        static_cast<double>(n);

    const double cy =
        static_cast<double>(
            window.sum_y)
        /
        static_cast<double>(n);

    result.x =
        static_cast<float>(cx);

    result.y =
        static_cast<float>(cy);

    result.sample_count =
        static_cast<std::uint32_t>(
            std::min<std::size_t>(
                n,
                0xFFFFFFFFu));

    result.newest_t =
        window.newest_t;

    radial_scratch_.clear();

    double sum_r = 0.0;

    // Constant for the whole snapshot; compute once, not per sample.
    const float histogram_max =
        std::max(
            1.0f,
            config_.histogram_max_radius_px);

    for (const CenterSample &sample :
         window.samples)
    {
        const double dx =
            static_cast<double>(
                sample.x) - cx;

        const double dy =
            static_cast<double>(
                sample.y) - cy;

        const float radius =
            static_cast<float>(
                std::sqrt(
                    dx * dx +
                    dy * dy));

        radial_scratch_.push_back(
            radius);

        sum_r += radius;

        std::size_t bin =
            static_cast<std::size_t>(
                radius
                /
                histogram_max
                *
                static_cast<float>(
                    RADIAL_HISTOGRAM_BINS));

        if (bin >= RADIAL_HISTOGRAM_BINS)
            bin = RADIAL_HISTOGRAM_BINS - 1;

        ++result.radial_histogram[bin];
    }

    const double mean_r =
        sum_r /
        static_cast<double>(n);

    result.mean_radius =
        static_cast<float>(
            mean_r);

    // Exact p95 without sorting the entire vector.
    const std::size_t p95_index =
        static_cast<std::size_t>(
            0.9 *
            static_cast<double>(
                n - 1));

    std::nth_element(
        radial_scratch_.begin(),
        radial_scratch_.begin()
            +
            static_cast<
                std::ptrdiff_t>(
                p95_index),
        radial_scratch_.end());

    result.p95_radius =
        radial_scratch_[p95_index];
    result.valid = true;
    
    return result;
}


void CenterWorker::publish_snapshot(
    std::uint32_t global_newest_t)
{
    CenterSnapshot snapshot;

    snapshot.window_us =
        config_.window_us;

    snapshot.newest_t =
        global_newest_t;

    snapshot.histogram_max_radius_px =
        config_.histogram_max_radius_px;

    for (std::size_t id = 0;
         id < CENTER_FREQ_COUNT;
         ++id)
    {
        snapshot.frequency[id] =
            compute_frequency_stats(
                windows_[id]);
    }

    store_.publish(snapshot);

    snapshots_published_.fetch_add(
        1,
        std::memory_order_relaxed);

    if (csv_.is_open()) {
        write_csv_snapshot(snapshot);
        maybe_flush_csv();
    }
}


void CenterWorker::open_csv() {
    if (config_.csv_path.empty())
        return;

    csv_.open(
        config_.csv_path,
        std::ios::out |
        std::ios::trunc);

    if (!csv_) {
        throw std::runtime_error(
            "Failed to open stats CSV: "
            + config_.csv_path);
    }

    // One row per frequency per stats snapshot.
    csv_
        << "timestamp_us,"
        << "window_us,"
        << "frequency_hz,"
        << "valid,"
        << "sample_count,"
        << "center_x_px,"
        << "center_y_px,"
        << "mean_radius_px,"
        << "p95_radius_px\n";

    last_csv_flush_ =
        std::chrono::steady_clock::now();
}


void CenterWorker::write_csv_snapshot(
    const CenterSnapshot &snapshot)
{
    static constexpr int frequency_hz[3] = {
        165, 366, 596
    };

    csv_
        << std::fixed
        << std::setprecision(3);

    for (std::size_t id = 0;
         id < CENTER_FREQ_COUNT;
         ++id)
    {
        const FrequencyCenter &s =
            snapshot.frequency[id];

        csv_
            << snapshot.newest_t
            << ','
            << snapshot.window_us
            << ','
            << frequency_hz[id]
            << ','
            << (s.valid ? 1 : 0)
            << ','
            << s.sample_count
            << ','
            << s.x
            << ','
            << s.y
            << ','
            << s.mean_radius
            << ','
            << s.p95_radius
            << '\n';

        csv_rows_written_.fetch_add(
            1,
            std::memory_order_relaxed);
    }
}


void CenterWorker::maybe_flush_csv() {
    if (!csv_.is_open())
        return;

    const auto now =
        std::chrono::steady_clock::now();

    const auto elapsed_ms =
        std::chrono::duration_cast<
            std::chrono::milliseconds>(
            now -
            last_csv_flush_)
            .count();

    if (
        elapsed_ms >=
        static_cast<long long>(
            config_.csv_flush_period_ms))
    {
        csv_.flush();

        last_csv_flush_ =
            now;
    }
}


void CenterWorker::run() {
    using Clock =
        std::chrono::steady_clock;

    std::uint32_t global_newest_t = 0;

    auto next_publish =
        Clock::now();

    while (
        !stop_requested_.load(
            std::memory_order_acquire))
    {
        const bool got_data =
            consume_available(
                global_newest_t);

        if (global_newest_t != 0)
            trim_windows(
                global_newest_t);

        const auto now =
            Clock::now();

        if (
            global_newest_t != 0
            &&
            now >= next_publish)
        {
            publish_snapshot(
                global_newest_t);

            next_publish =
                now
                +
                std::chrono::microseconds(
                    config_.update_period_us);
        }

        if (!got_data) {
            std::this_thread::sleep_for(
                std::chrono::microseconds(
                    200));
        }
    }

    consume_available(
        global_newest_t);

    if (global_newest_t != 0) {
        trim_windows(
            global_newest_t);

        publish_snapshot(
            global_newest_t);
    }
}

} // namespace event_led_pose
