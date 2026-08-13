#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <optional>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using TimestampUs = std::int64_t;

constexpr std::size_t kLedCount = 3;
constexpr double kMicrosecondsPerSecond = 1'000'000.0;

constexpr std::array<double, kLedCount> kFrequenciesHz{
    165.0,
    366.0,
    596.0,
};

double period_us(const std::size_t led_index)
{
    return kMicrosecondsPerSecond / kFrequenciesHz.at(led_index);
}

double wrap_positive(const double value, const double period)
{
    double result = std::fmod(value, period);
    if (result < 0.0) {
        result += period;
    }
    return result;
}

double wrap_centered(const double value, const double period)
{
    return value - period * std::floor(value / period + 0.5);
}

void require(const bool condition, const std::string_view message)
{
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

void run_formula_checks()
{
    constexpr int observation_count = 32;

    for (std::size_t j = 0; j < kLedCount; ++j) {
        const double T = period_us(j);
        const double phi = 0.37 * T;

        std::complex<double> Z{0.0, 0.0};
        double evidence = 0.0;

        for (int n = 0; n < observation_count; ++n) {
            const double t = phi + static_cast<double>(n) * T;
            const double angle =
                -2.0 * std::numbers::pi_v<double> * t / T;

            Z += std::polar(1.0, angle);
            evidence += 1.0;
        }

        const double estimated_phi = wrap_positive(
            -T * std::arg(Z) /
                (2.0 * std::numbers::pi_v<double>),
            T);

        const double phase_error =
            std::abs(wrap_centered(estimated_phi - phi, T));

        const double coherence =
            std::abs(Z) / (evidence + 1e-12);

        require(
            phase_error < 1e-8 * T,
            "Resonator phase-sign or dimensional check failed");

        require(
            coherence >= 0.0 && coherence <= 1.0 + 1e-12,
            "Coherence bound check failed");
    }

    constexpr double p_d = 0.90;
    constexpr double p_b = 0.01;

    const double hit_llr = std::log(p_d / p_b);
    const double miss_llr =
        std::log((1.0 - p_d) / (1.0 - p_b));

    require(hit_llr > 0.0, "Hit LLR has the wrong sign");
    require(miss_llr < 0.0, "Miss LLR has the wrong sign");

    constexpr double velocity_px_s = 500.0;
    constexpr TimestampUs delta_t_us = 2'000;

    const double displacement_px =
        velocity_px_s *
        static_cast<double>(delta_t_us) /
        kMicrosecondsPerSecond;

    require(
        std::abs(displacement_px - 1.0) < 1e-12,
        "Microsecond-to-second spatial conversion failed");
}

struct Event {
    std::uint16_t x{};
    std::uint16_t y{};
    TimestampUs t_us{};
    std::int8_t polarity{};
};

struct LabeledEvent {
    Event event{};
    int truth_led{-1};
};

enum class DecisionKind {
    Background,
    Unknown,
    Led,
};

struct Classification {
    DecisionKind kind{DecisionKind::Background};
    int led_index{-1};
    std::uint32_t candidate_id{};
    double association_cost{};
};

std::string_view decision_name(const Classification& result)
{
    if (result.kind == DecisionKind::Background) {
        return "background";
    }

    if (result.kind == DecisionKind::Unknown) {
        return "unknown";
    }

    switch (result.led_index) {
    case 0:
        return "led_165";
    case 1:
        return "led_366";
    case 2:
        return "led_596";
    default:
        return "invalid_led";
    }
}

enum class CandidateStage {
    Seed,
    Acquiring,
    Identified,
    Coasting,
};

struct Config {
    int sensor_width{1280};
    int sensor_height{720};
    int cell_size_px{16};
    int selected_polarity{1};

    TimestampUs microburst_width_us{100};
    int minimum_microburst_events{4};

    double birth_radius_px{28.0};
    double minimum_gate_radius_px{8.0};
    double acquisition_gate_radius_px{32.0};
    double maximum_gate_radius_px{90.0};
    double covariance_gate_scale{3.0};
    double gate_margin_px{4.0};

    double motion_uncertainty_px_s{250.0};
    double velocity_alpha{0.65};

    double minimum_timing_tolerance_us{55.0};
    double period_tolerance_fraction{0.020};
    double missed_cycle_tolerance_us{7.0};

    int required_matching_intervals{2};
    double required_coherence{0.70};
    double required_score_margin{0.65};

    int maximum_acquisition_gap_cycles{4};
    int maximum_missed_cycles{8};

    TimestampUs acquisition_timeout_us{35'000};
    double resonator_memory_us{80'000.0};

    double pll_alpha{0.35};
    double pll_beta{0.08};
    double maximum_period_error_fraction{0.02};

    double maximum_association_cost{2.0};
    double association_margin{0.15};

    std::size_t maximum_candidates{96};
};

struct RunningBurst {
    int count{};
    double mean_x{};
    double mean_y{};
    double mean_t_us{};
    double m2_x{};
    double m2_y{};

    void add(const Event& event)
    {
        ++count;

        const double n = static_cast<double>(count);

        const double dx = static_cast<double>(event.x) - mean_x;
        mean_x += dx / n;
        m2_x += dx * (static_cast<double>(event.x) - mean_x);

        const double dy = static_cast<double>(event.y) - mean_y;
        mean_y += dy / n;
        m2_y += dy * (static_cast<double>(event.y) - mean_y);

        const double dt = static_cast<double>(event.t_us) - mean_t_us;
        mean_t_us += dt / n;
    }

    double isotropic_sigma_px() const
    {
        if (count < 2) {
            return 1.0;
        }

        const double denominator =
            2.0 * static_cast<double>(count - 1);

        return std::sqrt(
            std::max(1.0, (m2_x + m2_y) / denominator));
    }
};

struct Candidate {
    bool active{};
    std::uint32_t generation{};

    CandidateStage stage{CandidateStage::Seed};
    int led_index{-1};

    double center_x{};
    double center_y{};
    double velocity_x_px_s{};
    double velocity_y_px_s{};
    double observed_sigma_px{2.0};

    int pulse_count{};
    int missed_cycles{};

    double epoch_us{};
    double last_observed_pulse_us{};
    double phase_time_us{};
    double estimated_period_us{};

    std::array<std::complex<double>, kLedCount> resonators{};
    std::array<double, kLedCount> evidence{};
    std::array<double, kLedCount> coherence{};
    std::array<int, kLedCount> matching_intervals{};
    std::array<double, kLedCount> interval_score{};

    bool burst_open{};
    RunningBurst burst{};
    TimestampUs burst_start_us{};
    std::uint32_t burst_token{};
    std::uint32_t expiry_token{};

    std::vector<int> indexed_cells;
};

struct ClassifierStats {
    std::uint64_t processed_events{};
    std::uint64_t polarity_rejections{};
    std::uint64_t background_decisions{};
    std::uint64_t unknown_decisions{};
    std::uint64_t led_decisions{};
    std::uint64_t candidate_evaluations{};
    std::uint64_t ambiguous_associations{};
    std::uint64_t candidates_created{};
    std::uint64_t candidates_retired{};
    std::uint64_t pulses_finalized{};

    std::size_t active_candidates{};
    std::size_t peak_active_candidates{};
    std::size_t peak_bucket_size{};
};

class SpatialIndex {
public:
    SpatialIndex(
        const int width,
        const int height,
        const int cell_size)
        : width_(width),
          height_(height),
          cell_size_(cell_size),
          columns_((width + cell_size - 1) / cell_size),
          rows_((height + cell_size - 1) / cell_size),
          buckets_(
              static_cast<std::size_t>(columns_ * rows_))
    {
    }

    const std::vector<std::size_t>& query(
        const int x,
        const int y) const
    {
        return buckets_.at(
            static_cast<std::size_t>(cell_index(x, y)));
    }

    void insert(
        const std::size_t candidate_slot,
        const int cell)
    {
        buckets_.at(static_cast<std::size_t>(cell))
            .push_back(candidate_slot);
    }

    void remove(
        const std::size_t candidate_slot,
        std::vector<int>& occupied_cells)
    {
        for (const int cell : occupied_cells) {
            auto& bucket =
                buckets_.at(static_cast<std::size_t>(cell));

            bucket.erase(
                std::remove(
                    bucket.begin(),
                    bucket.end(),
                    candidate_slot),
                bucket.end());
        }

        occupied_cells.clear();
    }

    int cell_from_coordinates(
        const int column,
        const int row) const
    {
        return row * columns_ + column;
    }

    int columns() const
    {
        return columns_;
    }

    int rows() const
    {
        return rows_;
    }

    int cell_size() const
    {
        return cell_size_;
    }

    std::size_t bucket_size(const int cell) const
    {
        return buckets_.at(static_cast<std::size_t>(cell)).size();
    }

private:
    int cell_index(const int x, const int y) const
    {
        const int column =
            std::clamp(x / cell_size_, 0, columns_ - 1);

        const int row =
            std::clamp(y / cell_size_, 0, rows_ - 1);

        return cell_from_coordinates(column, row);
    }

    int width_{};
    int height_{};
    int cell_size_{};
    int columns_{};
    int rows_{};

    std::vector<std::vector<std::size_t>> buckets_;
};

enum class DeadlineKind {
    BurstClose,
    CandidateExpiry,
};

struct Deadline {
    TimestampUs at_us{};
    std::size_t candidate_slot{};
    std::uint32_t candidate_generation{};
    std::uint32_t token{};
    DeadlineKind kind{};
};

struct DeadlineCompare {
    bool operator()(const Deadline& lhs, const Deadline& rhs) const
    {
        return lhs.at_us > rhs.at_us;
    }
};

class EventClassifier {
public:
    explicit EventClassifier(Config config = {})
        : config_(std::move(config)),
          index_(
              config_.sensor_width,
              config_.sensor_height,
              config_.cell_size_px),
          candidates_(config_.maximum_candidates)
    {
        led_owner_slots_.fill(-1);
    }

    Classification process(const Event& event)
    {
        advance_deadlines(event.t_us);
        advance_missed_pulses(event.t_us);

        ++stats_.processed_events;

        if (event.polarity != config_.selected_polarity) {
            ++stats_.polarity_rejections;
            return background();
        }

        const auto& nearby_candidates =
            index_.query(event.x, event.y);

        std::size_t best_slot =
            std::numeric_limits<std::size_t>::max();

        double best_cost = std::numeric_limits<double>::infinity();
        double second_cost = std::numeric_limits<double>::infinity();

        for (const std::size_t slot : nearby_candidates) {
            Candidate& candidate = candidates_.at(slot);

            if (!candidate.active) {
                continue;
            }

            ++stats_.candidate_evaluations;

            const auto temporal =
                temporal_normalized_error(candidate, event.t_us);

            if (!temporal.has_value()) {
                continue;
            }

            const auto [predicted_x, predicted_y] =
                predicted_center(candidate, event.t_us);

            const double radius =
                spatial_gate_radius(candidate, event.t_us);

            const double dx =
                static_cast<double>(event.x) - predicted_x;

            const double dy =
                static_cast<double>(event.y) - predicted_y;

            const double distance =
                std::sqrt(dx * dx + dy * dy);

            if (distance > radius) {
                continue;
            }

            const double spatial_normalized = distance / radius;

            const double cost =
                spatial_normalized * spatial_normalized +
                *temporal * *temporal;

            if (cost > config_.maximum_association_cost) {
                continue;
            }

            if (cost < best_cost) {
                second_cost = best_cost;
                best_cost = cost;
                best_slot = slot;
            } else if (cost < second_cost) {
                second_cost = cost;
            }
        }

        if (best_slot ==
            std::numeric_limits<std::size_t>::max()) {
            const auto seed = create_seed(event);

            if (!seed.has_value()) {
                return background();
            }

            return unknown(*seed, 0.0);
        }

        if (std::isfinite(second_cost) &&
            second_cost - best_cost <
                config_.association_margin) {
            ++stats_.ambiguous_associations;
            return unknown(0, best_cost);
        }

        Candidate& candidate = candidates_.at(best_slot);

        if (!candidate.burst_open) {
            start_burst(best_slot, event);
        } else {
            candidate.burst.add(event);
        }

        if (candidate.led_index >= 0 &&
            candidate.burst.count >=
                config_.minimum_microburst_events) {
            ++stats_.led_decisions;

            return Classification{
                .kind = DecisionKind::Led,
                .led_index = candidate.led_index,
                .candidate_id =
                    static_cast<std::uint32_t>(best_slot + 1),
                .association_cost = best_cost,
            };
        }

        return unknown(
            static_cast<std::uint32_t>(best_slot + 1),
            best_cost);
    }

    void flush(const TimestampUs through_us)
    {
        advance_deadlines(through_us);
        advance_missed_pulses(through_us);
    }

    const ClassifierStats& stats() const
    {
        return stats_;
    }

private:
    Classification background()
    {
        ++stats_.background_decisions;

        return Classification{
            .kind = DecisionKind::Background,
            .led_index = -1,
            .candidate_id = 0,
            .association_cost = 0.0,
        };
    }

    Classification unknown(
        const std::uint32_t candidate_id,
        const double cost)
    {
        ++stats_.unknown_decisions;

        return Classification{
            .kind = DecisionKind::Unknown,
            .led_index = -1,
            .candidate_id = candidate_id,
            .association_cost = cost,
        };
    }

    double timing_tolerance_us(
        const double period,
        const int cycles) const
    {
        return config_.minimum_timing_tolerance_us +
            config_.period_tolerance_fraction * period +
            config_.missed_cycle_tolerance_us *
                static_cast<double>(std::max(0, cycles - 1));
    }

    std::pair<double, double> predicted_center(
        const Candidate& candidate,
        const double timestamp_us) const
    {
        if (candidate.pulse_count == 0) {
            if (candidate.burst_open) {
                return {
                    candidate.burst.mean_x,
                    candidate.burst.mean_y,
                };
            }

            return {
                candidate.center_x,
                candidate.center_y,
            };
        }

        const double delta_t_s =
            std::max(
                0.0,
                timestamp_us -
                    candidate.last_observed_pulse_us) /
            kMicrosecondsPerSecond;

        return {
            candidate.center_x +
                candidate.velocity_x_px_s * delta_t_s,
            candidate.center_y +
                candidate.velocity_y_px_s * delta_t_s,
        };
    }

    double spatial_gate_radius(
        const Candidate& candidate,
        const double timestamp_us) const
    {
        if (candidate.pulse_count == 0) {
            return config_.birth_radius_px;
        }

        double base_radius =
            config_.covariance_gate_scale *
                candidate.observed_sigma_px +
            config_.gate_margin_px;

        base_radius =
            std::max(
                base_radius,
                config_.minimum_gate_radius_px);

        if (candidate.led_index < 0) {
            base_radius =
                std::max(
                    base_radius,
                    config_.acquisition_gate_radius_px);
        }

        const double delta_t_s =
            std::max(
                0.0,
                timestamp_us -
                    candidate.last_observed_pulse_us) /
            kMicrosecondsPerSecond;

        const double radius =
            base_radius +
            config_.motion_uncertainty_px_s * delta_t_s;

        return std::clamp(
            radius,
            config_.minimum_gate_radius_px,
            config_.maximum_gate_radius_px);
    }

    std::optional<double> temporal_normalized_error(
        const Candidate& candidate,
        const TimestampUs timestamp_us) const
    {
        if (candidate.burst_open) {
            if (timestamp_us <=
                candidate.burst_start_us +
                    config_.microburst_width_us) {
                return 0.0;
            }

            return std::nullopt;
        }

        if (candidate.pulse_count == 0) {
            return std::nullopt;
        }

        if (candidate.led_index >= 0) {
            const double delta =
                static_cast<double>(timestamp_us) -
                candidate.phase_time_us;

            const int cycles = static_cast<int>(
                std::llround(
                    delta / candidate.estimated_period_us));

            if (cycles < 1 ||
                cycles >
                    config_.maximum_missed_cycles + 1) {
                return std::nullopt;
            }

            const double predicted =
                candidate.phase_time_us +
                static_cast<double>(cycles) *
                    candidate.estimated_period_us;

            const double residual =
                static_cast<double>(timestamp_us) - predicted;

            const double tolerance =
                timing_tolerance_us(
                    candidate.estimated_period_us,
                    cycles);

            const double normalized =
                std::abs(residual) / tolerance;

            if (normalized > 1.0) {
                return std::nullopt;
            }

            return normalized;
        }

        const double interval =
            static_cast<double>(timestamp_us) -
            candidate.last_observed_pulse_us;

        double best =
            std::numeric_limits<double>::infinity();

        for (std::size_t j = 0; j < kLedCount; ++j) {
            const double T = period_us(j);

            const int cycles = static_cast<int>(
                std::llround(interval / T));

            if (cycles < 1 ||
                cycles >
                    config_.maximum_acquisition_gap_cycles) {
                continue;
            }

            const double residual =
                interval - static_cast<double>(cycles) * T;

            const double normalized =
                std::abs(residual) /
                timing_tolerance_us(T, cycles);

            best = std::min(best, normalized);
        }

        if (!std::isfinite(best) || best > 1.0) {
            return std::nullopt;
        }

        return best;
    }

    std::optional<std::size_t> create_seed(
        const Event& event)
    {
        for (std::size_t slot = 0;
             slot < candidates_.size();
             ++slot) {
            Candidate& candidate = candidates_[slot];

            if (candidate.active) {
                continue;
            }

            const std::uint32_t generation =
                candidate.generation + 1;

            candidate = Candidate{};
            candidate.active = true;
            candidate.generation =
                generation == 0 ? 1 : generation;
            candidate.stage = CandidateStage::Seed;
            candidate.center_x = event.x;
            candidate.center_y = event.y;

            ++stats_.candidates_created;
            ++stats_.active_candidates;

            stats_.peak_active_candidates =
                std::max(
                    stats_.peak_active_candidates,
                    stats_.active_candidates);

            start_burst(slot, event);
            index_candidate(slot, event.t_us);

            return slot;
        }

        return std::nullopt;
    }

    void start_burst(
        const std::size_t slot,
        const Event& event)
    {
        Candidate& candidate = candidates_.at(slot);

        candidate.burst = RunningBurst{};
        candidate.burst.add(event);
        candidate.burst_open = true;
        candidate.burst_start_us = event.t_us;
        ++candidate.burst_token;

        deadlines_.push(Deadline{
            .at_us =
                event.t_us + config_.microburst_width_us,
            .candidate_slot = slot,
            .candidate_generation = candidate.generation,
            .token = candidate.burst_token,
            .kind = DeadlineKind::BurstClose,
        });
    }

    void schedule_expiry(const std::size_t slot)
    {
        Candidate& candidate = candidates_.at(slot);

        ++candidate.expiry_token;

        deadlines_.push(Deadline{
            .at_us = static_cast<TimestampUs>(
                std::ceil(
                    candidate.last_observed_pulse_us +
                    static_cast<double>(
                        config_.acquisition_timeout_us))),
            .candidate_slot = slot,
            .candidate_generation = candidate.generation,
            .token = candidate.expiry_token,
            .kind = DeadlineKind::CandidateExpiry,
        });
    }

    void advance_deadlines(const TimestampUs now_us)
    {
        while (!deadlines_.empty() &&
               deadlines_.top().at_us <= now_us) {
            const Deadline deadline = deadlines_.top();
            deadlines_.pop();

            if (deadline.candidate_slot >=
                candidates_.size()) {
                continue;
            }

            Candidate& candidate =
                candidates_[deadline.candidate_slot];

            if (!candidate.active ||
                candidate.generation !=
                    deadline.candidate_generation) {
                continue;
            }

            if (deadline.kind == DeadlineKind::BurstClose) {
                if (!candidate.burst_open ||
                    candidate.burst_token !=
                        deadline.token) {
                    continue;
                }

                finalize_burst(deadline.candidate_slot);
                continue;
            }

            if (candidate.expiry_token != deadline.token ||
                candidate.led_index >= 0) {
                continue;
            }

            if (static_cast<double>(now_us) >=
                candidate.last_observed_pulse_us +
                    static_cast<double>(
                        config_.acquisition_timeout_us)) {
                deactivate_candidate(
                    deadline.candidate_slot);
            }
        }
    }

    void finalize_burst(const std::size_t slot)
    {
        Candidate& candidate = candidates_.at(slot);

        const RunningBurst completed = candidate.burst;
        candidate.burst = RunningBurst{};
        candidate.burst_open = false;

        if (completed.count <
            config_.minimum_microburst_events) {
            if (candidate.pulse_count == 0) {
                deactivate_candidate(slot);
            }

            return;
        }

        ++stats_.pulses_finalized;
        incorporate_pulse(slot, completed);
    }

    void incorporate_pulse(
        const std::size_t slot,
        const RunningBurst& pulse)
    {
        Candidate& candidate = candidates_.at(slot);

        const double pulse_t = pulse.mean_t_us;
        const double pulse_x = pulse.mean_x;
        const double pulse_y = pulse.mean_y;
        const double pulse_sigma =
            pulse.isotropic_sigma_px();

        if (candidate.pulse_count == 0) {
            candidate.epoch_us = pulse_t;
            candidate.last_observed_pulse_us = pulse_t;
            candidate.phase_time_us = pulse_t;
            candidate.center_x = pulse_x;
            candidate.center_y = pulse_y;
            candidate.observed_sigma_px = pulse_sigma;
            candidate.pulse_count = 1;
            candidate.stage = CandidateStage::Acquiring;

            for (std::size_t j = 0; j < kLedCount; ++j) {
                candidate.resonators[j] = {1.0, 0.0};
                candidate.evidence[j] = 1.0;
                candidate.coherence[j] =
                    1.0 / (1.0 + 1e-12);
            }

            schedule_expiry(slot);
            index_candidate(
                slot,
                static_cast<TimestampUs>(
                    std::llround(pulse_t)));
            return;
        }

        const double previous_t =
            candidate.last_observed_pulse_us;

        const double interval_us = pulse_t - previous_t;

        if (interval_us <= 0.0) {
            return;
        }

        const double interval_s =
            interval_us / kMicrosecondsPerSecond;

        const double measured_velocity_x =
            (pulse_x - candidate.center_x) / interval_s;

        const double measured_velocity_y =
            (pulse_y - candidate.center_y) / interval_s;

        candidate.velocity_x_px_s =
            (1.0 - config_.velocity_alpha) *
                candidate.velocity_x_px_s +
            config_.velocity_alpha *
                measured_velocity_x;

        candidate.velocity_y_px_s =
            (1.0 - config_.velocity_alpha) *
                candidate.velocity_y_px_s +
            config_.velocity_alpha *
                measured_velocity_y;

        for (std::size_t j = 0; j < kLedCount; ++j) {
            const double T = period_us(j);

            const double decay =
                std::exp(
                    -interval_us /
                    config_.resonator_memory_us);

            const double relative_t =
                pulse_t - candidate.epoch_us;

            const double angle =
                -2.0 *
                std::numbers::pi_v<double> *
                relative_t / T;

            candidate.resonators[j] =
                decay * candidate.resonators[j] +
                std::polar(1.0, angle);

            candidate.evidence[j] =
                decay * candidate.evidence[j] + 1.0;

            candidate.coherence[j] =
                std::abs(candidate.resonators[j]) /
                (candidate.evidence[j] + 1e-12);

            const int cycles = static_cast<int>(
                std::llround(interval_us / T));

            bool matched = false;
            double normalized = 0.0;

            if (cycles >= 1 &&
                cycles <=
                    config_.maximum_missed_cycles + 1) {
                const double residual =
                    interval_us -
                    static_cast<double>(cycles) * T;

                normalized =
                    std::abs(residual) /
                    timing_tolerance_us(T, cycles);

                matched = normalized <= 1.0;
            }

            if (matched) {
                ++candidate.matching_intervals[j];

                candidate.interval_score[j] +=
                    1.0 -
                    0.5 * normalized * normalized;
            } else {
                candidate.interval_score[j] -= 0.25;
            }
        }

        if (candidate.led_index >= 0) {
            const int cycles = std::max(
                1,
                static_cast<int>(
                    std::llround(
                        (pulse_t -
                         candidate.phase_time_us) /
                        candidate.estimated_period_us)));

            const double predicted =
                candidate.phase_time_us +
                static_cast<double>(cycles) *
                    candidate.estimated_period_us;

            const double residual = pulse_t - predicted;

            candidate.phase_time_us =
                predicted +
                config_.pll_alpha * residual;

            candidate.estimated_period_us +=
                config_.pll_beta *
                residual /
                static_cast<double>(cycles);

            const double nominal =
                period_us(
                    static_cast<std::size_t>(
                        candidate.led_index));

            candidate.estimated_period_us =
                std::clamp(
                    candidate.estimated_period_us,
                    nominal *
                        (1.0 -
                         config_
                             .maximum_period_error_fraction),
                    nominal *
                        (1.0 +
                         config_
                             .maximum_period_error_fraction));

            candidate.missed_cycles = 0;
            candidate.stage = CandidateStage::Identified;
        } else {
            candidate.phase_time_us = pulse_t;
        }

        candidate.last_observed_pulse_us = pulse_t;
        candidate.center_x = pulse_x;
        candidate.center_y = pulse_y;

        candidate.observed_sigma_px =
            0.70 * candidate.observed_sigma_px +
            0.30 * pulse_sigma;

        ++candidate.pulse_count;

        if (candidate.led_index < 0) {
            try_identify(slot);
        }

        if (!candidate.active) {
            return;
        }

        if (candidate.led_index < 0) {
            schedule_expiry(slot);
        }

        index_candidate(
            slot,
            static_cast<TimestampUs>(
                std::llround(pulse_t)));
    }

    double frequency_score(
        const Candidate& candidate,
        const std::size_t led_index) const
    {
        return
            static_cast<double>(
                candidate.matching_intervals[led_index]) +
            candidate.coherence[led_index] +
            0.15 * candidate.interval_score[led_index];
    }

    void try_identify(const std::size_t slot)
    {
        Candidate& candidate = candidates_.at(slot);

        std::array<int, kLedCount> order{0, 1, 2};

        std::sort(
            order.begin(),
            order.end(),
            [&](const int lhs, const int rhs) {
                return frequency_score(
                           candidate,
                           static_cast<std::size_t>(lhs)) >
                    frequency_score(
                           candidate,
                           static_cast<std::size_t>(rhs));
            });

        const int best = order[0];
        const int second = order[1];

        const double best_score =
            frequency_score(
                candidate,
                static_cast<std::size_t>(best));

        const double second_score =
            frequency_score(
                candidate,
                static_cast<std::size_t>(second));

        if (candidate.matching_intervals[
                static_cast<std::size_t>(best)] <
                config_.required_matching_intervals ||
            candidate.coherence[
                static_cast<std::size_t>(best)] <
                config_.required_coherence ||
            best_score - second_score <
                config_.required_score_margin) {
            return;
        }

        const int existing_slot =
            led_owner_slots_[
                static_cast<std::size_t>(best)];

        if (existing_slot >= 0 &&
            candidates_[
                static_cast<std::size_t>(existing_slot)]
                .active) {
            const Candidate& existing =
                candidates_[
                    static_cast<std::size_t>(
                        existing_slot)];

            const double existing_score =
                frequency_score(
                    existing,
                    static_cast<std::size_t>(best));

            if (best_score <= existing_score + 0.25) {
                deactivate_candidate(slot);
                return;
            }

            deactivate_candidate(
                static_cast<std::size_t>(
                    existing_slot));
        }

        candidate.led_index = best;
        candidate.stage = CandidateStage::Identified;
        candidate.missed_cycles = 0;
        candidate.estimated_period_us =
            period_us(static_cast<std::size_t>(best));
        candidate.phase_time_us =
            candidate.last_observed_pulse_us;

        led_owner_slots_[
            static_cast<std::size_t>(best)] =
            static_cast<int>(slot);
    }

    void advance_missed_pulses(
        const TimestampUs now_us)
    {
        for (std::size_t led = 0;
             led < kLedCount;
             ++led) {
            const int owner = led_owner_slots_[led];

            if (owner < 0) {
                continue;
            }

            const std::size_t slot =
                static_cast<std::size_t>(owner);

            Candidate& candidate = candidates_.at(slot);

            if (!candidate.active ||
                candidate.led_index !=
                    static_cast<int>(led)) {
                led_owner_slots_[led] = -1;
                continue;
            }

            if (candidate.burst_open) {
                continue;
            }

            while (candidate.active) {
                const int next_cycle =
                    candidate.missed_cycles + 1;

                const double expected =
                    candidate.phase_time_us +
                    static_cast<double>(next_cycle) *
                        candidate.estimated_period_us;

                const double tolerance =
                    timing_tolerance_us(
                        candidate.estimated_period_us,
                        next_cycle);

                if (static_cast<double>(now_us) <=
                    expected + tolerance) {
                    break;
                }

                ++candidate.missed_cycles;
                candidate.stage = CandidateStage::Coasting;

                if (candidate.missed_cycles >
                    config_.maximum_missed_cycles) {
                    deactivate_candidate(slot);
                    break;
                }

                index_candidate(slot, now_us);
            }
        }
    }

    void index_candidate(
        const std::size_t slot,
        const TimestampUs now_us)
    {
        Candidate& candidate = candidates_.at(slot);

        index_.remove(
            slot,
            candidate.indexed_cells);

        if (!candidate.active) {
            return;
        }

        double horizon_us =
            2.0 * period_us(0);

        if (candidate.led_index >= 0) {
            horizon_us = std::max(
                10'000.0,
                2.0 * candidate.estimated_period_us);
        }

        const auto [x0, y0] =
            predicted_center(candidate, now_us);

        const auto [x1, y1] =
            predicted_center(
                candidate,
                static_cast<double>(now_us) +
                    horizon_us);

        const double radius = std::max(
            spatial_gate_radius(candidate, now_us),
            spatial_gate_radius(
                candidate,
                static_cast<double>(now_us) +
                    horizon_us));

        const double min_x =
            std::min(x0, x1) - radius;

        const double max_x =
            std::max(x0, x1) + radius;

        const double min_y =
            std::min(y0, y1) - radius;

        const double max_y =
            std::max(y0, y1) + radius;

        const int cell_size = index_.cell_size();

        const int column0 = std::clamp(
            static_cast<int>(
                std::floor(min_x / cell_size)),
            0,
            index_.columns() - 1);

        const int column1 = std::clamp(
            static_cast<int>(
                std::floor(max_x / cell_size)),
            0,
            index_.columns() - 1);

        const int row0 = std::clamp(
            static_cast<int>(
                std::floor(min_y / cell_size)),
            0,
            index_.rows() - 1);

        const int row1 = std::clamp(
            static_cast<int>(
                std::floor(max_y / cell_size)),
            0,
            index_.rows() - 1);

        for (int row = row0; row <= row1; ++row) {
            for (int column = column0;
                 column <= column1;
                 ++column) {
                const int cell =
                    index_.cell_from_coordinates(
                        column,
                        row);

                index_.insert(slot, cell);
                candidate.indexed_cells.push_back(cell);

                stats_.peak_bucket_size =
                    std::max(
                        stats_.peak_bucket_size,
                        index_.bucket_size(cell));
            }
        }
    }

    void deactivate_candidate(const std::size_t slot)
    {
        Candidate& candidate = candidates_.at(slot);

        if (!candidate.active) {
            return;
        }

        index_.remove(
            slot,
            candidate.indexed_cells);

        if (candidate.led_index >= 0) {
            const std::size_t led =
                static_cast<std::size_t>(
                    candidate.led_index);

            if (led_owner_slots_[led] ==
                static_cast<int>(slot)) {
                led_owner_slots_[led] = -1;
            }
        }

        candidate.active = false;

        ++stats_.candidates_retired;

        if (stats_.active_candidates > 0) {
            --stats_.active_candidates;
        }
    }

    Config config_;
    SpatialIndex index_;
    std::vector<Candidate> candidates_;

    std::array<int, kLedCount> led_owner_slots_{};

    std::priority_queue<
        Deadline,
        std::vector<Deadline>,
        DeadlineCompare>
        deadlines_;

    ClassifierStats stats_{};
};

struct ScenarioConfig {
    int sensor_width{1280};
    int sensor_height{720};
    TimestampUs duration_us{1'200'000};
    std::uint32_t seed{7};

    double background_events_per_second{12'000.0};

    bool include_scale_change{true};
    bool include_occlusion{true};
    bool include_moving_distractor{true};
};

std::pair<double, double> led_center(
    const int led,
    const double time_fraction)
{
    const double two_pi =
        2.0 * std::numbers::pi_v<double>;

    switch (led) {
    case 0:
        return {
            220.0 + 300.0 * time_fraction,
            190.0 +
                35.0 *
                    std::sin(two_pi * time_fraction),
        };

    case 1:
        return {
            620.0 +
                80.0 *
                    std::sin(two_pi * time_fraction),
            340.0 - 100.0 * time_fraction,
        };

    default:
        return {
            1'000.0 - 260.0 * time_fraction,
            220.0 +
                60.0 *
                    std::sin(2.0 * two_pi *
                             time_fraction),
        };
    }
}

bool is_occluded(
    const int led,
    const double time_us)
{
    switch (led) {
    case 0:
        return time_us >= 420'000.0 &&
            time_us <= 456'000.0;

    case 1:
        return time_us >= 620'000.0 &&
            time_us <= 637'000.0;

    case 2:
        return time_us >= 820'000.0 &&
            time_us <= 830'000.0;

    default:
        return false;
    }
}

std::vector<LabeledEvent> generate_scenario(
    const ScenarioConfig& config)
{
    std::mt19937_64 generator(config.seed);

    std::normal_distribution<double> timestamp_jitter_us{
        0.0,
        8.0,
    };

    std::vector<LabeledEvent> events;

    for (std::size_t led = 0; led < kLedCount; ++led) {
        const double T = period_us(led);
        const double phase_us =
            1'000.0 + 700.0 * static_cast<double>(led);

        for (std::uint64_t cycle = 0;; ++cycle) {
            const double transition_us =
                phase_us +
                static_cast<double>(cycle) * T;

            if (transition_us >=
                static_cast<double>(config.duration_us)) {
                break;
            }

            if (config.include_occlusion &&
                is_occluded(
                    static_cast<int>(led),
                    transition_us)) {
                continue;
            }

            const double fraction =
                transition_us /
                static_cast<double>(config.duration_us);

            const auto [center_x, center_y] =
                led_center(
                    static_cast<int>(led),
                    fraction);

            const double sigma_px =
                config.include_scale_change
                ? 2.0 + 6.0 * fraction
                : 3.0;

            const int event_count =
                18 +
                static_cast<int>(
                    std::lround(2.0 * sigma_px));

            std::normal_distribution<double> spatial_x{
                center_x,
                sigma_px,
            };

            std::normal_distribution<double> spatial_y{
                center_y,
                sigma_px,
            };

            for (int i = 0; i < event_count; ++i) {
                const TimestampUs event_time =
                    std::clamp<TimestampUs>(
                        static_cast<TimestampUs>(
                            std::llround(
                                transition_us +
                                timestamp_jitter_us(
                                    generator))),
                        0,
                        config.duration_us - 1);

                const int x = std::clamp(
                    static_cast<int>(
                        std::lround(spatial_x(generator))),
                    0,
                    config.sensor_width - 1);

                const int y = std::clamp(
                    static_cast<int>(
                        std::lround(spatial_y(generator))),
                    0,
                    config.sensor_height - 1);

                events.push_back(LabeledEvent{
                    .event = Event{
                        .x = static_cast<std::uint16_t>(x),
                        .y = static_cast<std::uint16_t>(y),
                        .t_us = event_time,
                        .polarity = 1,
                    },
                    .truth_led = static_cast<int>(led),
                });
            }
        }
    }

    const double duration_s =
        static_cast<double>(config.duration_us) /
        kMicrosecondsPerSecond;

    std::poisson_distribution<int> background_count{
        config.background_events_per_second *
        duration_s,
    };

    std::uniform_int_distribution<TimestampUs> random_time{
        0,
        config.duration_us - 1,
    };

    std::uniform_int_distribution<int> random_x{
        0,
        config.sensor_width - 1,
    };

    std::uniform_int_distribution<int> random_y{
        0,
        config.sensor_height - 1,
    };

    std::bernoulli_distribution random_polarity{0.5};

    const int number_of_background_events =
        background_count(generator);

    for (int i = 0;
         i < number_of_background_events;
         ++i) {
        events.push_back(LabeledEvent{
            .event = Event{
                .x = static_cast<std::uint16_t>(
                    random_x(generator)),
                .y = static_cast<std::uint16_t>(
                    random_y(generator)),
                .t_us = random_time(generator),
                .polarity = static_cast<std::int8_t>(
                    random_polarity(generator)
                        ? 1
                        : -1),
            },
            .truth_led = -1,
        });
    }

    if (config.include_moving_distractor) {
        std::uniform_real_distribution<double>
            distractor_interval_us{
                1'100.0,
                9'000.0,
            };

        std::normal_distribution<double>
            distractor_jitter_us{
                0.0,
                12.0,
            };

        double time_us = 5'000.0;

        while (time_us <
               static_cast<double>(config.duration_us)) {
            const double fraction =
                time_us /
                static_cast<double>(config.duration_us);

            const double center_x =
                100.0 + 900.0 * fraction;

            const double center_y =
                600.0 - 350.0 * fraction;

            std::normal_distribution<double> x_distribution{
                center_x,
                4.0,
            };

            std::normal_distribution<double> y_distribution{
                center_y,
                4.0,
            };

            for (int i = 0; i < 12; ++i) {
                const TimestampUs event_time =
                    std::clamp<TimestampUs>(
                        static_cast<TimestampUs>(
                            std::llround(
                                time_us +
                                distractor_jitter_us(
                                    generator))),
                        0,
                        config.duration_us - 1);

                const int x = std::clamp(
                    static_cast<int>(
                        std::lround(
                            x_distribution(generator))),
                    0,
                    config.sensor_width - 1);

                const int y = std::clamp(
                    static_cast<int>(
                        std::lround(
                            y_distribution(generator))),
                    0,
                    config.sensor_height - 1);

                events.push_back(LabeledEvent{
                    .event = Event{
                        .x =
                            static_cast<std::uint16_t>(x),
                        .y =
                            static_cast<std::uint16_t>(y),
                        .t_us = event_time,
                        .polarity = 1,
                    },
                    .truth_led = -1,
                });
            }

            time_us +=
                distractor_interval_us(generator);
        }
    }

    std::sort(
        events.begin(),
        events.end(),
        [](const LabeledEvent& lhs,
           const LabeledEvent& rhs) {
            if (lhs.event.t_us != rhs.event.t_us) {
                return lhs.event.t_us < rhs.event.t_us;
            }

            if (lhs.event.x != rhs.event.x) {
                return lhs.event.x < rhs.event.x;
            }

            return lhs.event.y < rhs.event.y;
        });

    return events;
}

struct Metrics {
    std::uint64_t total_events{};
    std::uint64_t led_events{};
    std::uint64_t background_events{};

    std::uint64_t correctly_labeled{};
    std::uint64_t incorrectly_labeled{};
    std::uint64_t rejected_led_events{};
    std::uint64_t background_labeled_as_led{};

    std::uint64_t post_warmup_led_events{};
    std::uint64_t post_warmup_correct{};

    std::array<TimestampUs, kLedCount> first_lock_us{
        -1,
        -1,
        -1,
    };

    double precision{};
    double post_warmup_recall{};
    double background_false_positive_rate{};
    double processing_mevents_per_second{};
};

Metrics evaluate(
    const std::vector<LabeledEvent>& events,
    const std::vector<Classification>& results,
    const double processing_seconds,
    const TimestampUs warmup_us)
{
    require(
        events.size() == results.size(),
        "Result count does not match event count");

    Metrics metrics;
    metrics.total_events = events.size();

    for (std::size_t i = 0; i < events.size(); ++i) {
        const int truth = events[i].truth_led;
        const Classification& result = results[i];

        if (truth >= 0) {
            ++metrics.led_events;

            if (events[i].event.t_us >= warmup_us) {
                ++metrics.post_warmup_led_events;
            }

            if (result.kind == DecisionKind::Led &&
                result.led_index == truth) {
                ++metrics.correctly_labeled;

                if (events[i].event.t_us >= warmup_us) {
                    ++metrics.post_warmup_correct;
                }

                TimestampUs& first_lock =
                    metrics.first_lock_us[
                        static_cast<std::size_t>(truth)];

                if (first_lock < 0) {
                    first_lock = events[i].event.t_us;
                }
            } else if (result.kind == DecisionKind::Led) {
                ++metrics.incorrectly_labeled;
            } else {
                ++metrics.rejected_led_events;
            }
        } else {
            ++metrics.background_events;

            if (result.kind == DecisionKind::Led) {
                ++metrics.background_labeled_as_led;
            }
        }
    }

    const double led_output_count =
        static_cast<double>(
            metrics.correctly_labeled +
            metrics.incorrectly_labeled +
            metrics.background_labeled_as_led);

    metrics.precision =
        led_output_count > 0.0
        ? static_cast<double>(
              metrics.correctly_labeled) /
              led_output_count
        : 0.0;

    metrics.post_warmup_recall =
        metrics.post_warmup_led_events > 0
        ? static_cast<double>(
              metrics.post_warmup_correct) /
              static_cast<double>(
                  metrics.post_warmup_led_events)
        : 0.0;

    metrics.background_false_positive_rate =
        metrics.background_events > 0
        ? static_cast<double>(
              metrics.background_labeled_as_led) /
              static_cast<double>(
                  metrics.background_events)
        : 0.0;

    metrics.processing_mevents_per_second =
        processing_seconds > 0.0
        ? static_cast<double>(events.size()) /
              processing_seconds /
              1'000'000.0
        : 0.0;

    return metrics;
}

void write_csv(
    const std::string& filename,
    const std::vector<LabeledEvent>& events,
    const std::vector<Classification>& results)
{
    std::ofstream output(filename);

    if (!output) {
        throw std::runtime_error(
            "Cannot open CSV output: " + filename);
    }

    output
        << "t_us,x,y,polarity,truth_led,decision,"
           "candidate_id,association_cost\n";

    output << std::setprecision(10);

    for (std::size_t i = 0; i < events.size(); ++i) {
        output
            << events[i].event.t_us << ','
            << events[i].event.x << ','
            << events[i].event.y << ','
            << static_cast<int>(
                   events[i].event.polarity)
            << ','
            << events[i].truth_led << ','
            << decision_name(results[i]) << ','
            << results[i].candidate_id << ','
            << results[i].association_cost
            << '\n';
    }
}

bool passes_default_acceptance(const Metrics& metrics)
{
    const bool every_led_locked =
        std::all_of(
            metrics.first_lock_us.begin(),
            metrics.first_lock_us.end(),
            [](const TimestampUs timestamp) {
                return timestamp >= 0;
            });

    return every_led_locked &&
        metrics.precision >= 0.98 &&
        metrics.post_warmup_recall >= 0.70 &&
        metrics.background_false_positive_rate <= 0.005;
}

void print_report(
    const Metrics& metrics,
    const ClassifierStats& stats)
{
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "\nFrequency configuration\n";

    for (std::size_t j = 0; j < kLedCount; ++j) {
        std::cout
            << "  LED " << j
            << ": " << kFrequenciesHz[j] << " Hz"
            << ", period=" << period_us(j)
            << " us\n";
    }

    std::cout << "\nLock times\n";

    for (std::size_t j = 0; j < kLedCount; ++j) {
        std::cout
            << "  " << kFrequenciesHz[j] << " Hz: ";

        if (metrics.first_lock_us[j] < 0) {
            std::cout << "not locked\n";
        } else {
            std::cout
                << static_cast<double>(
                       metrics.first_lock_us[j]) /
                       1'000.0
                << " ms\n";
        }
    }

    std::cout
        << "\nClassification metrics\n"
        << "  total events: "
        << metrics.total_events << '\n'
        << "  LED events: "
        << metrics.led_events << '\n'
        << "  background events: "
        << metrics.background_events << '\n'
        << "  correctly labeled: "
        << metrics.correctly_labeled << '\n'
        << "  incorrectly labeled: "
        << metrics.incorrectly_labeled << '\n'
        << "  rejected/unknown LED events: "
        << metrics.rejected_led_events << '\n'
        << "  background labeled as LED: "
        << metrics.background_labeled_as_led << '\n'
        << "  precision: "
        << metrics.precision << '\n'
        << "  post-warmup recall: "
        << metrics.post_warmup_recall << '\n'
        << "  background false-positive rate: "
        << metrics.background_false_positive_rate << '\n'
        << "  classifier throughput: "
        << metrics.processing_mevents_per_second
        << " Mevent/s\n";

    const double evaluations_per_event =
        stats.processed_events > 0
        ? static_cast<double>(
              stats.candidate_evaluations) /
              static_cast<double>(
                  stats.processed_events)
        : 0.0;

    std::cout
        << "\nClassifier structure metrics\n"
        << "  candidates created: "
        << stats.candidates_created << '\n'
        << "  candidates retired: "
        << stats.candidates_retired << '\n'
        << "  peak active candidates: "
        << stats.peak_active_candidates << '\n'
        << "  peak grid-bucket size: "
        << stats.peak_bucket_size << '\n'
        << "  candidate evaluations/event: "
        << evaluations_per_event << '\n'
        << "  finalized microbursts: "
        << stats.pulses_finalized << '\n'
        << "  ambiguous associations: "
        << stats.ambiguous_associations << '\n';
}

struct CommandLine {
    ScenarioConfig scenario{};
    std::optional<std::string> csv_filename;
};

void print_usage(const char* executable)
{
    std::cout
        << "Usage: " << executable << " [options]\n\n"
        << "Options:\n"
        << "  --csv FILE             Write every event decision\n"
        << "  --seed N               Random seed\n"
        << "  --duration-ms N        Scenario duration\n"
        << "  --background-rate N    Background events/second\n"
        << "  --no-occlusion         Disable LED occlusions\n"
        << "  --no-scale-change      Disable LED scale growth\n"
        << "  --no-distractor        Disable moving distractor\n"
        << "  --help                  Show this message\n";
}

CommandLine parse_command_line(
    const int argc,
    char** argv)
{
    CommandLine command_line;

    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];

        auto value_after = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(
                    "Missing value after " + argument);
            }

            return argv[++i];
        };

        if (argument == "--csv") {
            command_line.csv_filename = value_after();
        } else if (argument == "--seed") {
            command_line.scenario.seed =
                static_cast<std::uint32_t>(
                    std::stoul(value_after()));
        } else if (argument == "--duration-ms") {
            command_line.scenario.duration_us =
                static_cast<TimestampUs>(
                    std::stoll(value_after()) * 1'000);
        } else if (argument == "--background-rate") {
            command_line
                .scenario
                .background_events_per_second =
                std::stod(value_after());
        } else if (argument == "--no-occlusion") {
            command_line.scenario.include_occlusion = false;
        } else if (argument == "--no-scale-change") {
            command_line
                .scenario
                .include_scale_change = false;
        } else if (argument == "--no-distractor") {
            command_line
                .scenario
                .include_moving_distractor = false;
        } else if (argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error(
                "Unknown argument: " + argument);
        }
    }

    return command_line;
}

} // namespace

int main(const int argc, char** argv)
{
    try {
        const CommandLine command_line =
            parse_command_line(argc, argv);

        run_formula_checks();

        std::cout << "Formula checks: PASS\n";

        const std::vector<LabeledEvent> events =
            generate_scenario(command_line.scenario);

        EventClassifier classifier;
        std::vector<Classification> results;
        results.reserve(events.size());

        const auto start =
            std::chrono::steady_clock::now();

        for (const LabeledEvent& labeled_event : events) {
            results.push_back(
                classifier.process(labeled_event.event));
        }

        if (!events.empty()) {
            classifier.flush(
                events.back().event.t_us + 200);
        }

        const auto finish =
            std::chrono::steady_clock::now();

        const double processing_seconds =
            std::chrono::duration<double>(
                finish - start)
                .count();

        const Metrics metrics =
            evaluate(
                events,
                results,
                processing_seconds,
                100'000);

        if (command_line.csv_filename.has_value()) {
            write_csv(
                *command_line.csv_filename,
                events,
                results);

            std::cout
                << "CSV written to "
                << *command_line.csv_filename
                << '\n';
        }

        print_report(metrics, classifier.stats());

        const bool passed =
            passes_default_acceptance(metrics);

        std::cout
            << "\nSynthetic acceptance: "
            << (passed ? "PASS" : "FAIL")
            << '\n';

        return passed ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr
            << "error: "
            << error.what()
            << '\n';

        return 1;
    }
}