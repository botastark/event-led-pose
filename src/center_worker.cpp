#include "center_worker.hpp"

#include <algorithm>

namespace event_led_pose {

void CenterStore::publish(const CenterSnapshot &snapshot) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshot_ = snapshot;
    }
    version_.fetch_add(1, std::memory_order_release);
}

bool CenterStore::copy_if_new(std::uint64_t &last_version,
                              CenterSnapshot &out) const {
    const std::uint64_t version =
        version_.load(std::memory_order_acquire);

    if (version == last_version)
        return false;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        out = snapshot_;
    }

    last_version = version;
    return true;
}

CenterWorker::CenterWorker(const Config &config,
                           CenterStore &store)
    : config_(config),
      store_(store),
      input_(config.input_ring_capacity) {}

CenterWorker::~CenterWorker() {
    stop();
}

void CenterWorker::start() {
    if (thread_.joinable())
        return;

    stop_requested_.store(false, std::memory_order_release);
    thread_ = std::thread(&CenterWorker::run, this);
}

void CenterWorker::stop() {
    stop_requested_.store(true, std::memory_order_release);

    if (thread_.joinable())
        thread_.join();
}

void CenterWorker::begin_batch() noexcept {
    batch_submitted_ = 0;
    batch_dropped_ = 0;
    input_.producer_begin();
}

bool CenterWorker::push(std::uint16_t x,
                        std::uint16_t y,
                        std::uint32_t t,
                        std::uint8_t frequency_id) noexcept {
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

bool CenterWorker::consume_available(std::uint32_t &global_newest_t) {
    std::uint64_t tail = input_.consumer_tail();
    const std::uint64_t head = input_.consumer_head();

    if (head == tail)
        return false;

    const auto wall_now =
        std::chrono::steady_clock::now();

    for (std::uint64_t seq = tail; seq < head; ++seq) {
        const CenterSample &sample =
            input_.consumer_at(seq);

        global_newest_t = sample.t;

        auto &window =
            windows_[sample.frequency_id];

        window.samples.push_back(sample);
        window.sum_x += sample.x;
        window.sum_y += sample.y;
        window.newest_t = sample.t;
        window.last_wall_sample = wall_now;

        while (window.samples.size() >
               config_.max_samples_per_frequency) {
            const auto old = window.samples.front();
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

void CenterWorker::trim_windows(std::uint32_t global_newest_t) {
    for (auto &window : windows_) {
        while (!window.samples.empty()) {
            const std::uint32_t age =
                global_newest_t -
                window.samples.front().t;

            if (age <= config_.window_us)
                break;

            const auto old =
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

void CenterWorker::publish_snapshot() {
    CenterSnapshot snapshot;
    snapshot.window_us = config_.window_us;

    const auto now =
        std::chrono::steady_clock::now();

    for (std::size_t id = 0;
         id < CENTER_FREQ_COUNT;
         ++id) {
        const auto &window =
            windows_[id];

        auto &center =
            snapshot.frequency[id];

        const std::size_t n =
            window.samples.size();

        if (n < config_.min_samples)
            continue;

        if (window.last_wall_sample.time_since_epoch().count() == 0)
            continue;

        const auto stale_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now - window.last_wall_sample).count();

        if (stale_ms >
            static_cast<long long>(config_.wall_stale_ms))
            continue;

        center.valid = true;
        center.sample_count =
            static_cast<std::uint32_t>(
                std::min<std::size_t>(n, 0xFFFFFFFFu));

        center.x =
            static_cast<float>(
                static_cast<double>(window.sum_x) /
                static_cast<double>(n));

        center.y =
            static_cast<float>(
                static_cast<double>(window.sum_y) /
                static_cast<double>(n));

        center.newest_t =
            window.newest_t;
    }

    store_.publish(snapshot);
}

void CenterWorker::run() {
    using Clock = std::chrono::steady_clock;

    std::uint32_t global_newest_t = 0;

    auto next_publish =
        Clock::now();

    while (!stop_requested_.load(
        std::memory_order_acquire)) {
        const bool got_data =
            consume_available(global_newest_t);

        if (global_newest_t != 0)
            trim_windows(global_newest_t);

        const auto now =
            Clock::now();

        if (now >= next_publish) {
            publish_snapshot();

            next_publish =
                now +
                std::chrono::microseconds(
                    config_.update_period_us);
        }

        if (!got_data) {
            std::this_thread::sleep_for(
                std::chrono::microseconds(200));
        }
    }

    consume_available(global_newest_t);

    if (global_newest_t != 0)
        trim_windows(global_newest_t);

    publish_snapshot();
}

} // namespace event_led_pose
