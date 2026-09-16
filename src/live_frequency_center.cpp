#include "fast_event_viewer.hpp"
#include "center_worker.hpp"
#include "packed_event.hpp"
#include "spsc_ring.hpp"

#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/stream/camera.h>
#include <metavision/hal/facilities/i_ll_biases.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <regex>
#include <sstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using event_led_pose::PackedEvent;
using event_led_pose::pack_event;
using event_led_pose::event_x;
using event_led_pose::event_y;
using event_led_pose::event_p;

// ============================================================
// TARGET FREQUENCIES
// ============================================================
//
// Positive event  = OFF -> ON transition
// Negative event  = ON  -> OFF transition
//
// Same transition type repeats every full LED period T:
//
// OFF->ON ... OFF->ON   ~= n*T
// ON ->OFF ... ON ->OFF ~= n*T
//
// Opposite transition types should be separated by:
//
// OFF->ON ... ON->OFF ~= (n + 0.5)*T
//
// We use same-type full-period evidence as the primary classifier.
// Opposite-type half-period evidence is optional confirmation.
//

struct FrequencyConfig {
    std::uint16_t period_us;
    std::array<std::uint16_t, 4> expected;
    std::array<std::uint16_t, 4> tolerance;
    const char *name;
};

// ~ +/-2% windows. Missed cycles allowed up to 4T.
constexpr std::array<FrequencyConfig, 3> FREQ = {{

    // 165 Hz: ~ +/-1%
{6061,
 {6061,12121,0,0},
 { 61,121,0,0},
 "165 Hz"},

    // 366 Hz: ~ +/-1.5%
    {2732,
     {2732,5464,8197,10929},
     { 41, 82,123,164},
     "366 Hz"},

    // 596 Hz: ~ +/-1.5%
    {1678,
     {1678,3356,5034,6711},
     { 25, 50, 76,101},
     "596 Hz"}

}};
constexpr std::uint8_t NO_CANDIDATE = 0xFF;

// Same-polarity burst suppression.
constexpr std::uint32_t BURST_MERGE_US = 300;

// Intervals larger than this cannot be represented by uint16_t.
// Our largest useful interval is 4T_165 ~= 24.2 ms, so uint16_t
// is more than enough and saves memory.
constexpr std::uint32_t MAX_STORED_DT_US = 25000;

// Candidate timing.
constexpr std::uint32_t CANDIDATE_TIMEOUT_US = 20000;

// Opposite-transition half-period tolerance.
constexpr std::uint32_t CROSS_TOLERANCE_PERMILLE = 10; // 1%

// Robustness thresholds:
//
// Strong case:
//   at least one valid OFF->ON full-period interval AND
//   at least one valid ON->OFF full-period interval.
//
// Fallback case:
//   2 valid full-period intervals from one transition direction
//   when the other polarity is weak/missing.


// TINY PER-POLARITY INTERVAL RING
// ============================================================
//
// Two full-period observations per transition direction.
// Combined pixel history:
//   2 x OFF->ON periods
//   2 x ON->OFF periods
//
// Stored as uint16_t because all relevant dt are < 30 ms.
//
// This is much smaller than storing 4-6 absolute uint32 timestamps
// per polarity.
//

struct IntervalRing2 {
    std::uint16_t dt0 = 0;
    std::uint16_t dt1 = 0;

    // newest first; with N=2 a shift is cheaper than maintaining
    // a separate ring index.
    inline void push(std::uint16_t dt) noexcept {
        dt1 = dt0;
        dt0 = dt;
    }

    inline int count_nonzero() const noexcept {
        return (dt0 != 0) + (dt1 != 0);
    }
};


// ============================================================
// PER-PIXEL STATE
// ============================================================
//
// 20 bytes/pixel on normal 4-byte alignment:
//
//   last OFF->ON timestamp             4
//   last ON->OFF timestamp             4
//   2 OFF->ON period intervals         4
//   2 ON->OFF period intervals         4
//   candidate/confidence/flags         4
//
// At 1280x720 this is ~17.6 MiB.
//
// Compare that with two 6-timestamp uint32 histories:
//   12 timestamps * 4 bytes ~= 48 bytes/pixel before metadata.
//

struct PixelState {
    std::uint32_t last_rise = 0; // OFF -> ON (positive)
    std::uint32_t last_fall = 0; // ON  -> OFF (negative)

    IntervalRing2 rise_periods;
    IntervalRing2 fall_periods;

    // Coarse candidate-support timestamp in 256 us ticks.
    // Keeps PixelState at 20 bytes while making timeout real.
    std::uint16_t last_support_tick = 0;

    std::uint8_t candidate = NO_CANDIDATE;

    // bit 0: candidate has rise support
    // bit 1: candidate has fall support
    std::uint8_t support_mask = 0;
};

static_assert(sizeof(PixelState) == 20,
              "PixelState size changed; memory budget affected");

constexpr std::uint32_t CANDIDATE_TICK_SHIFT = 8;
constexpr std::uint32_t CANDIDATE_TICK_US =
    1u << CANDIDATE_TICK_SHIFT;

constexpr std::uint16_t CANDIDATE_TIMEOUT_TICKS =
    static_cast<std::uint16_t>(
        (CANDIDATE_TIMEOUT_US + CANDIDATE_TICK_US - 1u)
        / CANDIDATE_TICK_US);

inline std::uint16_t support_tick(
    std::uint32_t t) noexcept
{
    return static_cast<std::uint16_t>(
        t >> CANDIDATE_TICK_SHIFT);
}

inline void reset_candidate(
    PixelState &s) noexcept
{
    s.rise_periods = {};
    s.fall_periods = {};
    s.last_support_tick = 0;
    s.candidate = NO_CANDIDATE;
    s.support_mask = 0;
}

inline bool candidate_timed_out(
    const PixelState &s,
    std::uint32_t now) noexcept
{
    if (s.candidate == NO_CANDIDATE ||
        s.last_support_tick == 0)
    {
        return false;
    }

    const std::uint16_t now_tick =
        support_tick(now);

    const std::uint16_t age_ticks =
        static_cast<std::uint16_t>(
            now_tick - s.last_support_tick);

    return age_ticks > CANDIDATE_TIMEOUT_TICKS;
}

struct Result {
    bool passed = false;
    std::uint8_t id = NO_CANDIDATE;
};


// ============================================================
// INTEGER MATCHING
// ============================================================

inline bool match_same_transition_period(
    std::uint32_t dt,
    const FrequencyConfig &f) noexcept
{
    // Compiler can unroll this fixed loop.
    for (int i = 0; i < 4; ++i) {
        const std::uint32_t e =
            f.expected[static_cast<std::size_t>(i)];

        if (e == 0)
            continue;

        const std::uint32_t tol =
            f.tolerance[static_cast<std::size_t>(i)];

        if (dt >= e - tol && dt <= e + tol)
            return true;
    }

    return false;
}


// Cheap acquisition gate.
// Returns the best frequency whose n*T window contains dt.
inline std::uint8_t classify_interval(
    std::uint32_t dt) noexcept
{
    std::uint8_t best = NO_CANDIDATE;

    std::uint32_t best_error = 0xFFFFFFFFu;
    std::uint32_t best_expected = 1;

    for (std::uint8_t i = 0; i < 3; ++i) {
        const auto &f = FREQ[i];

        for (int k = 0; k < 4; ++k) {
            const std::uint32_t e =
                f.expected[static_cast<std::size_t>(k)];

            if (e == 0)
                continue;

            const std::uint32_t tol =
                f.tolerance[static_cast<std::size_t>(k)];

            if (dt < e - tol || dt > e + tol)
                continue;

            const std::uint32_t err =
                dt > e ? dt - e : e - dt;

            // Compare relative error without floating point.
            if (best == NO_CANDIDATE ||
                static_cast<std::uint64_t>(err) * best_expected <
                static_cast<std::uint64_t>(best_error) * e)
            {
                best = i;
                best_error = err;
                best_expected = e;
            }
        }
    }

    return best;
}


// Opposite transition phase:
//
//   2*dt ~= odd*T
//
// because dt ~= 0.5T, 1.5T, 2.5T, ...
inline bool match_cross_transition_phase(
    std::uint32_t dt,
    std::uint16_t period) noexcept
{
    const std::uint64_t twice =
        2ull * static_cast<std::uint64_t>(dt);

    static constexpr std::array<std::uint32_t, 8> ODD = {
        1,3,5,7,9,11,13,15
    };

    for (const std::uint32_t n : ODD) {
        const std::uint64_t expected =
            static_cast<std::uint64_t>(n) * period;

        const std::uint64_t error =
            twice > expected
                ? twice - expected
                : expected - twice;

        if (error * 1000ull <=
            expected * CROSS_TOLERANCE_PERMILLE)
        {
            return true;
        }
    }

    return false;
}


inline int ring_matches(const IntervalRing2 &ring,
                        const FrequencyConfig &f) noexcept {
    int matches = 0;

    if (ring.dt0 != 0 &&
        match_same_transition_period(ring.dt0, f))
        ++matches;

    if (ring.dt1 != 0 &&
        match_same_transition_period(ring.dt1, f))
        ++matches;

    return matches;
}


// Score a frequency using BOTH transition directions.
//
// Primary evidence:
//   OFF->ON -> OFF->ON = nT
//   ON->OFF -> ON->OFF = nT
//
// Strong lock:
//   at least one matching interval from both transition directions.
//
// Fallback:
//   three total matches if one polarity is not reliable.
//
// Cross-phase can confirm but is not mandatory.
inline int score_frequency(const PixelState &s,
                           std::uint8_t id,
                           std::uint32_t now,
                           bool current_polarity) noexcept {
    const auto &f = FREQ[id];

    const int rise =
        ring_matches(s.rise_periods, f);

    const int fall =
        ring_matches(s.fall_periods, f);

    int score = rise + fall;

    if (rise > 0 && fall > 0)
        score += 2; // explicit both-edge bonus

    // Optional cross-transition phase confirmation using newest
    // rise/fall absolute timestamps.
    if (s.last_rise != 0 && s.last_fall != 0) {
        const std::uint32_t dt =
            s.last_rise > s.last_fall
                ? s.last_rise - s.last_fall
                : s.last_fall - s.last_rise;

        if (match_cross_transition_phase(dt, f.period_us))
            ++score;
    }

    (void)now;
    (void)current_polarity;

    return score;
}


// ============================================================
// TRANSITION-RING FILTER
// ============================================================

inline Result process_event(PixelState &s,
                            bool polarity,
                            std::uint32_t now) noexcept {
    Result out;

    // Actually enforce candidate timeout before using this event.
    if (candidate_timed_out(s, now)) {
        reset_candidate(s);
    }

    // Positive polarity = OFF -> ON (rise)
    // Negative polarity = ON  -> OFF (fall)
    std::uint32_t &last_same =
        polarity ? s.last_rise : s.last_fall;

    IntervalRing2 &period_ring =
        polarity ? s.rise_periods : s.fall_periods;

    const std::uint8_t support_bit =
        polarity ? 0x01 : 0x02;

    // --------------------------------------------------------
    // 1. BURST SUPPRESSION
    // --------------------------------------------------------

    if (last_same != 0) {
        const std::uint32_t dt =
            now - last_same;

        if (dt < BURST_MERGE_US)
            return out;
    }

    // --------------------------------------------------------
    // 2. SAME-TRANSITION PERIOD
    // --------------------------------------------------------

    const std::uint32_t previous_same =
        last_same;

    last_same = now;

    if (previous_same == 0)
        return out;

    const std::uint32_t dt =
        now - previous_same;

    if (dt > MAX_STORED_DT_US)
        return out;

    // First use the active candidate if there is one.
    // This is the fast steady-state path.
    if (s.candidate != NO_CANDIDATE) {
        const auto &f =
            FREQ[s.candidate];

        if (match_same_transition_period(dt, f)) {
            period_ring.push(
                static_cast<std::uint16_t>(dt));

            s.support_mask |= support_bit;

            // Refresh lifetime only on genuine matching evidence.
            s.last_support_tick =
                support_tick(now);

            const int score =
                score_frequency(
                    s,
                    s.candidate,
                    now,
                    polarity);

            (void)score;

            const int rise =
                ring_matches(
                    s.rise_periods,
                    f);

            const int fall =
                ring_matches(
                    s.fall_periods,
                    f);

            const int total =
                rise + fall;

            const bool both_edges =
                rise > 0 && fall > 0;

            const bool robust =
                rise >= 2 &&
                fall >= 2;

            if (robust) {
                out.passed = true;
                out.id = s.candidate;
            }

            return out;
        }

        // Candidate did not match this same-transition interval.
        //
        // Break temporal coherence for THIS polarity immediately.
        // This prevents isolated matches separated by arbitrary
        // background intervals from accumulating in Ring2.
        period_ring = {};
        s.support_mask &=
            static_cast<std::uint8_t>(~support_bit);

        // See whether this interval points to a different known
        // frequency.
        const std::uint8_t other =
            classify_interval(dt);

        if (other != NO_CANDIDATE &&
            other != s.candidate)
        {
            s.rise_periods = {};
            s.fall_periods = {};

            s.candidate = other;
            s.support_mask = support_bit;
            s.last_support_tick =
                support_tick(now);

            period_ring.push(
                static_cast<std::uint16_t>(dt));

            return out;
        }

        // If no coherent support remains, release the candidate now.
        if (s.rise_periods.count_nonzero() == 0 &&
            s.fall_periods.count_nonzero() == 0)
        {
            reset_candidate(s);
        }

        return out;
    }

    // --------------------------------------------------------
    // 3. ACQUISITION
    // --------------------------------------------------------
    //
    // No candidate:
    // use one same-transition interval to find a plausible known
    // frequency. Random unmatched intervals are NOT stored.
    //

    const std::uint8_t id =
        classify_interval(dt);

    if (id == NO_CANDIDATE)
        return out;

    s.candidate = id;
    s.support_mask = support_bit;
    s.last_support_tick =
        support_tick(now);

    s.rise_periods = {};
    s.fall_periods = {};

    period_ring.push(
        static_cast<std::uint16_t>(dt));

    return out;
}


// ============================================================
// CLI / PARALLEL PIPELINE
// ============================================================

struct Options {
    bool visualize = false;
    bool show_raw = false;

    unsigned display_fps = 120;
    std::uint32_t persistence_us = 6000;

    // freshness bound for worker input
    std::uint64_t max_filter_backlog_events = 100000;

    std::size_t filter_ring_capacity = 1u << 19;

    // Center estimation is a separate lossy diagnostic path.
    // Only every Nth classified event is copied to it.
    std::uint32_t center_stride = 4;

    // Sliding event-time window for live center + radial stats.
    // Shorter windows reduce motion-smear in the measured distribution.
    std::uint32_t center_window_us = 10000;

    // Publish live stats at 100 Hz by default.
    std::uint32_t center_update_us = 10000;

    // Hard bound per frequency for close-up/high-rate LEDs.
    std::size_t center_max_samples = 20000;

    // Optional buffered CSV output for offline motion analysis.
    // Empty = no recording.
    std::string stats_csv_path;

    // Shared radial histogram X-axis for all three LEDs.
    float histogram_max_radius_px = 400.0f;

    bool show_histograms = true;

    // Custom JSON file with a nested "biases" object.
    // Example:
    // {
    //   "biases": {
    //     "bias_diff_on": 140,
    //     "bias_diff_off": 190,
    //     "bias_refr": 55,
    //     "bias_fo": 55,
    //     "bias_hpf": 0
    //   }
    // }
    std::string bias_config_path;

    bool print_biases = false;
};

Options parse_args(int argc, char **argv) {
    Options o;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];

        if (a == "--visualize") {
            o.visualize = true;
        }
        else if (a == "--show-raw") {
            o.show_raw = true;
        }
        else if (a == "--display-fps") {
            if (++i >= argc)
                throw std::runtime_error("--display-fps needs value");

            o.display_fps =
                static_cast<unsigned>(
                    std::stoul(argv[i]));
        }
        else if (a == "--persistence-us") {
            if (++i >= argc)
                throw std::runtime_error("--persistence-us needs value");

            o.persistence_us =
                static_cast<std::uint32_t>(
                    std::stoul(argv[i]));
        }
        else if (a == "--max-filter-backlog") {
            if (++i >= argc)
                throw std::runtime_error("--max-filter-backlog needs value");

            o.max_filter_backlog_events =
                std::stoull(argv[i]);
        }
        else if (a == "--center-stride") {
            if (++i >= argc)
                throw std::runtime_error("--center-stride needs value");

            o.center_stride =
                static_cast<std::uint32_t>(std::stoul(argv[i]));

            if (o.center_stride == 0)
                throw std::runtime_error("--center-stride must be >= 1");
        }
        else if (a == "--center-window-us") {
            if (++i >= argc)
                throw std::runtime_error("--center-window-us needs value");

            o.center_window_us =
                static_cast<std::uint32_t>(std::stoul(argv[i]));
        }
        else if (a == "--center-update-us") {
            if (++i >= argc)
                throw std::runtime_error("--center-update-us needs value");

            o.center_update_us =
                static_cast<std::uint32_t>(std::stoul(argv[i]));
        }
        else if (a == "--center-max-samples") {
            if (++i >= argc)
                throw std::runtime_error("--center-max-samples needs value");

            o.center_max_samples =
                static_cast<std::size_t>(std::stoull(argv[i]));
        }
        else if (a == "--stats-csv") {
            if (++i >= argc)
                throw std::runtime_error("--stats-csv needs path");

            o.stats_csv_path = argv[i];
        }
        else if (a == "--hist-max-radius") {
            if (++i >= argc)
                throw std::runtime_error("--hist-max-radius needs value");

            o.histogram_max_radius_px =
                std::stof(argv[i]);

            if (o.histogram_max_radius_px <= 0.0f)
                throw std::runtime_error("--hist-max-radius must be > 0");
        }
        else if (a == "--no-histograms") {
            o.show_histograms = false;
        }
        else if (a == "--bias-config") {
            if (++i >= argc)
                throw std::runtime_error("--bias-config needs path");

            o.bias_config_path = argv[i];
        }
        else if (a == "--print-biases") {
            o.print_biases = true;
        }
        else if (a == "--help" || a == "-h") {
            std::cout
                << "Usage: live_frequency_center [options]\n"
                << "  --visualize\n"
                << "  --show-raw\n"
                << "  --display-fps N\n"
                << "  --persistence-us N\n"
                << "  --max-filter-backlog N\n"
                << "  --center-stride N\n"
                << "  --center-window-us N\n"
                << "  --center-update-us N\n"
                << "  --center-max-samples N\n"
                << "  --stats-csv FILE.csv\n"
                << "  --hist-max-radius PX\n"
                << "  --no-histograms\n"
                << "  --bias-config FILE.json\n"
                << "  --print-biases\n";

            std::exit(0);
        }
        else {
            throw std::runtime_error(
                "Unknown argument: " + a);
        }
    }

    return o;
}


struct BiasConfig {
    int bias_diff_on = 0;
    int bias_diff_off = 0;
    int bias_refr = 0;
    int bias_fo = 0;
    int bias_hpf = 0;
};

std::string read_text_file(const std::string &path) {
    std::ifstream input(path);

    if (!input) {
        throw std::runtime_error(
            "Could not open bias config: " + path);
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

int parse_required_json_int(const std::string &json,
                            const char *key) {
    // The input format is intentionally small and fixed. We search for an
    // integer value associated with a quoted key rather than introducing a
    // JSON dependency in the low-latency executable.
    const std::regex pattern(
        std::string("\\\"") + key +
        "\\\"\\s*:\\s*(-?[0-9]+)");

    std::smatch match;

    if (!std::regex_search(json, match, pattern)) {
        throw std::runtime_error(
            std::string("Missing integer bias '") +
            key + "' in bias config");
    }

    return std::stoi(match[1].str());
}

BiasConfig load_bias_config(const std::string &path) {
    const std::string json = read_text_file(path);

    // Require the expected nested section so that a random JSON file with
    // similarly named keys is not silently accepted.
    if (json.find("\"biases\"") == std::string::npos) {
        throw std::runtime_error(
            "Bias config does not contain a 'biases' object: " + path);
    }

    BiasConfig config;
    config.bias_diff_on =
        parse_required_json_int(json, "bias_diff_on");
    config.bias_diff_off =
        parse_required_json_int(json, "bias_diff_off");
    config.bias_refr =
        parse_required_json_int(json, "bias_refr");
    config.bias_fo =
        parse_required_json_int(json, "bias_fo");
    config.bias_hpf =
        parse_required_json_int(json, "bias_hpf");

    return config;
}

void print_current_biases(Metavision::Camera &camera) {
    auto &biases =
        camera.get_facility<Metavision::I_LL_Biases>();

    static constexpr const char *NAMES[] = {
        "bias_diff_on",
        "bias_diff_off",
        "bias_refr",
        "bias_fo",
        "bias_hpf"
    };

    std::cout << "Active camera biases:\n";

    for (const char *name : NAMES) {
        std::cout
            << "  "
            << name
            << " = "
            << biases.get(name)
            << "\n";
    }
}

void apply_bias_config(Metavision::Camera &camera,
                       const BiasConfig &config) {
    auto &biases =
        camera.get_facility<Metavision::I_LL_Biases>();

    const auto set_bias =
        [&](const char *name, int requested) {
            if (!biases.set(name, requested)) {
                throw std::runtime_error(
                    std::string("Failed to set ") +
                    name + "=" +
                    std::to_string(requested));
            }

            const int active = biases.get(name);

            std::cout
                << "  "
                << name
                << ": requested="
                << requested
                << " active="
                << active
                << "\n";
        };

    std::cout << "Applying sensor biases:\n";

    set_bias("bias_diff_on",  config.bias_diff_on);
    set_bias("bias_diff_off", config.bias_diff_off);
    set_bias("bias_refr",     config.bias_refr);
    set_bias("bias_fo",       config.bias_fo);
    set_bias("bias_hpf",      config.bias_hpf);
}

} // namespace


int main(int argc, char **argv) {
    using namespace Metavision;
    using event_led_pose::FastEventViewer;
    using event_led_pose::SpscRing;
    using event_led_pose::CenterStore;
    using event_led_pose::CenterWorker;

    try {
        const Options options =
            parse_args(argc, argv);

        Camera camera =
            Camera::from_first_available();

        // Biases are applied before camera.start(), so irrelevant sensor
        // activity can be reduced before events reach the host filter.
        if (!options.bias_config_path.empty()) {
            const BiasConfig bias_config =
                load_bias_config(
                    options.bias_config_path);

            std::cout
                << "Loaded bias config: "
                << options.bias_config_path
                << "\n";

            apply_bias_config(
                camera,
                bias_config);
        }

        if (options.print_biases) {
            print_current_biases(camera);
        }

        const auto &geometry =
            camera.geometry();

        const int width =
            geometry.get_width();

        const int height =
            geometry.get_height();

        std::cout
            << "Camera: "
            << width << " x " << height
            << "\n"
            << "Pixel state: "
            << sizeof(PixelState)
            << " bytes/pixel (~"
            << (
                sizeof(PixelState)
                * static_cast<double>(width)
                * static_cast<double>(height)
                / (1024.0 * 1024.0)
               )
            << " MiB)\n"
            << "Filter model:\n"
            << "  OFF->ON to OFF->ON = n*T\n"
            << "  ON->OFF to ON->OFF = n*T\n"
            << "  opposite edge phase ~= (n+0.5)*T bonus\n";

        // ----------------------------------------------------
        // TRANSPORT RING:
        //
        // producer = camera callback
        // consumer = frequency worker
        //
        // This ring ONLY moves packed events between threads.
        // It is not the frequency-history ring.
        // ----------------------------------------------------

        SpscRing<PackedEvent>
            filter_input(
                options.filter_ring_capacity);

        // ----------------------------------------------------
        // CLASSIFIED EVENTS -> SPATIAL STATS WORKER
        //
        // This is a separate lossy path. The frequency worker only
        // copies sampled accepted events into an SPSC queue.
        // Center/radius statistics and CSV writing happen here,
        // never in the latency-critical filter thread.
        // ----------------------------------------------------

        CenterStore center_store;

        CenterWorker::Config center_cfg;
        center_cfg.window_us = options.center_window_us;
        center_cfg.update_period_us = options.center_update_us;
        center_cfg.max_samples_per_frequency =
            options.center_max_samples;

        center_cfg.histogram_max_radius_px =
            options.histogram_max_radius_px;

        center_cfg.csv_path =
            options.stats_csv_path;

        CenterWorker center_worker(
            center_cfg,
            center_store);

        center_worker.start();

        std::unique_ptr<FastEventViewer>
            viewer;

        if (options.visualize) {
            FastEventViewer::Config cfg;

            cfg.sensor_width = width;
            cfg.sensor_height = height;
            cfg.max_fps = options.display_fps;
            cfg.persistence_us = options.persistence_us;

            cfg.title =
                options.show_raw
                ? "Raw + frequency + spatial stats"
                : "Frequency + spatial stats";

            viewer =
                std::make_unique<
                    FastEventViewer>(
                        cfg,
                        &center_store);
        }

        std::atomic<bool> stop{false};

        std::atomic<std::uint32_t>
            newest_camera_ts{0};

        std::atomic<std::uint32_t>
            newest_filtered_ts{0};

        std::atomic<std::uint64_t>
            camera_events{0};

        std::atomic<std::uint64_t>
            filter_input_drops{0};

        std::atomic<std::uint64_t>
            filter_stale_skips{0};

        std::atomic<std::uint64_t>
            passed_events{0};

        std::array<
            std::atomic<std::uint64_t>, 3>
            freq_counts{};

        // ----------------------------------------------------
        // FILTER WORKER
        // ----------------------------------------------------

        std::thread filter_thread([&] {
            std::vector<PixelState> pixels(
                static_cast<std::size_t>(width)
                * static_cast<std::size_t>(height));

            std::uint64_t local_passed = 0;
            std::array<std::uint64_t, 3>
                local_counts{};

            std::uint64_t local_stale_skips = 0;
            std::uint64_t center_counter = 0;

            while (!stop.load(
                std::memory_order_acquire))
            {
                std::uint64_t tail =
                    filter_input.consumer_tail();

                const std::uint64_t head =
                    filter_input.consumer_head();

                if (head == tail) {
                    std::this_thread::sleep_for(
                        std::chrono::microseconds(50));
                    continue;
                }

                std::uint64_t backlog =
                    head - tail;

                // Freshness over completeness:
                // never replay seconds of old events.
                if (backlog >
                    options.max_filter_backlog_events)
                {
                    const std::uint64_t skip =
                        backlog -
                        options.max_filter_backlog_events;

                    tail += skip;
                    local_stale_skips += skip;
                }

                if (viewer)
                    viewer->freq_begin_batch();

                center_worker.begin_batch();

                for (std::uint64_t seq = tail;
                     seq < head;
                     ++seq)
                {
                    const PackedEvent &e =
                        filter_input.consumer_at(seq);

                    const std::uint16_t x =
                        event_x(e);

                    const std::uint16_t y =
                        event_y(e);

                    const bool p =
                        event_p(e);

                    const std::uint32_t t =
                        e.t;

                    PixelState &state =
                        pixels[
                            static_cast<std::size_t>(y)
                            * static_cast<std::size_t>(width)
                            + static_cast<std::size_t>(x)
                        ];

                    const Result r =
                        process_event(
                            state,
                            p,
                            t);

                    if (r.passed) {
                        ++local_passed;
                        ++local_counts[r.id];

                        if (viewer) {
                            viewer->freq_push(
                                x, y, t, r.id);
                        }

                        ++center_counter;

                        if ((center_counter %
                             options.center_stride) == 0u)
                        {
                            center_worker.push(
                                x,
                                y,
                                t,
                                r.id);
                        }
                    }

                    newest_filtered_ts.store(
                        t,
                        std::memory_order_relaxed);
                }

                if (viewer)
                    viewer->freq_end_batch();

                center_worker.end_batch();

                filter_input.consumer_commit(head);
            }

            passed_events.store(
                local_passed,
                std::memory_order_relaxed);

            filter_stale_skips.store(
                local_stale_skips,
                std::memory_order_relaxed);

            for (std::size_t i = 0; i < 3; ++i) {
                freq_counts[i].store(
                    local_counts[i],
                    std::memory_order_relaxed);
            }
        });

        // ----------------------------------------------------
        // CAMERA CALLBACK
        // ----------------------------------------------------
        //
        // No frequency filtering here.
        // Only pack and publish events.
        // ----------------------------------------------------

        camera.cd().add_callback(
            [&](const EventCD *begin,
                const EventCD *end)
            {
                if (begin == end)
                    return;

                camera_events.fetch_add(
                    static_cast<std::uint64_t>(
                        end - begin),
                    std::memory_order_relaxed);

                filter_input.producer_begin();

                if (viewer &&
                    options.show_raw)
                {
                    viewer->raw_begin_batch();
                }

                std::uint64_t local_drops = 0;

                for (const EventCD *ev = begin;
                     ev != end;
                     ++ev)
                {
                    const PackedEvent pe =
                        pack_event(
                            ev->x,
                            ev->y,
                            ev->p,
                            static_cast<
                                std::uint32_t>(
                                ev->t));

                    if (!filter_input
                            .producer_push(pe))
                    {
                        ++local_drops;
                    }

                    if (viewer &&
                        options.show_raw)
                    {
                        viewer->raw_push(pe);
                    }
                }

                filter_input.producer_end();

                if (viewer &&
                    options.show_raw)
                {
                    viewer->raw_end_batch();
                }

                if (local_drops != 0) {
                    filter_input_drops.fetch_add(
                        local_drops,
                        std::memory_order_relaxed);
                }

                newest_camera_ts.store(
                    static_cast<std::uint32_t>(
                        (end - 1)->t),
                    std::memory_order_relaxed);
            });

        camera.start();

        if (viewer) {
            std::cout
                << "Parallel transition-ring filter + center worker running.\n"
                << "Q/Esc closes viewer.\n";

            viewer->run();
        }
        else {
            std::cout
                << "Parallel transition-ring filter + center worker running.\n"
                << "Press ENTER to stop.\n";

            std::cin.get();
        }

        camera.stop();

        stop.store(
            true,
            std::memory_order_release);

        filter_thread.join();

        center_worker.stop();

        const std::uint32_t cam_t =
            newest_camera_ts.load();

        const std::uint32_t fil_t =
            newest_filtered_ts.load();

        std::cout
            << "\n============================\n"
            << "RESULTS\n"
            << "============================\n"
            << "Camera events      : "
            << camera_events.load()
            << "\n"
            << "Transport drops    : "
            << filter_input_drops.load()
            << "\n"
            << "Stale filter skips : "
            << filter_stale_skips.load()
            << "\n"
            << "Passed events      : "
            << passed_events.load()
            << "\n"
            << "Final filter lag   : "
            << (cam_t - fil_t) / 1000.0
            << " ms\n";

        for (std::size_t i = 0; i < 3; ++i) {
            std::cout
                << "  "
                << FREQ[i].name
                << " : "
                << freq_counts[i].load()
                << "\n";
        }

        std::cout
            << "\nSpatial stats worker\n"
            << "  submitted       : "
            << center_worker.submitted()
            << "\n"
            << "  input drops     : "
            << center_worker.input_drops()
            << "\n"
            << "  window drops    : "
            << center_worker.window_drops()
            << "\n"
            << "  snapshots       : "
            << center_worker.snapshots_published()
            << "\n"
            << "  CSV rows        : "
            << center_worker.csv_rows_written()
            << "\n";

        if (!options.stats_csv_path.empty()) {
            std::cout
                << "  CSV file        : "
                << options.stats_csv_path
                << "\n";
        }

        return 0;
    }
    catch (const std::exception &e) {
        std::cerr
            << "Error: "
            << e.what()
            << "\n";

        return 1;
    }
}
