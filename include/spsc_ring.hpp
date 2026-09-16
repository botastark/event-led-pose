#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace event_led_pose {

// Fixed-size, single-producer / single-consumer ring.
// Producer publishes one whole camera batch with one release-store.
// Consumer may intentionally skip old entries to preserve freshness.
template<typename T>
class SpscRing {
public:
    explicit SpscRing(std::size_t capacity)
        : buffer_(capacity), mask_(capacity - 1) {
        if (capacity == 0 || (capacity & (capacity - 1)) != 0)
            throw std::invalid_argument("SpscRing capacity must be power of two");
    }

    SpscRing(const SpscRing &) = delete;
    SpscRing &operator=(const SpscRing &) = delete;

    // ---------- producer ----------
    inline void producer_begin() noexcept {
        prod_head_ = head_.load(std::memory_order_relaxed);
        const std::uint64_t tail = tail_.load(std::memory_order_acquire);
        prod_limit_ = tail + static_cast<std::uint64_t>(buffer_.size());
        prod_open_ = true;
    }

    inline bool producer_push(const T &v) noexcept {
        if (!prod_open_)
            producer_begin();

        if (prod_head_ >= prod_limit_)
            return false;

        buffer_[static_cast<std::size_t>(prod_head_) & mask_] = v;
        ++prod_head_;
        return true;
    }

    inline void producer_end() noexcept {
        if (!prod_open_)
            return;
        head_.store(prod_head_, std::memory_order_release);
        prod_open_ = false;
    }

    // ---------- consumer ----------
    inline std::uint64_t consumer_head() const noexcept {
        return head_.load(std::memory_order_acquire);
    }

    inline std::uint64_t consumer_tail() const noexcept {
        return tail_.load(std::memory_order_relaxed);
    }

    inline const T &consumer_at(std::uint64_t seq) const noexcept {
        return buffer_[static_cast<std::size_t>(seq) & mask_];
    }

    inline void consumer_commit(std::uint64_t tail) noexcept {
        tail_.store(tail, std::memory_order_release);
    }

    inline std::size_t capacity() const noexcept {
        return buffer_.size();
    }

private:
    std::vector<T> buffer_;
    std::size_t mask_;

    alignas(64) std::atomic<std::uint64_t> head_{0};
    alignas(64) std::atomic<std::uint64_t> tail_{0};

    // producer-thread only
    std::uint64_t prod_head_ = 0;
    std::uint64_t prod_limit_ = 0;
    bool prod_open_ = false;
};

} // namespace event_led_pose
