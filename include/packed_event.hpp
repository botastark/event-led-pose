#pragma once
#include <cstdint>

namespace event_led_pose {

struct PackedEvent {
    std::uint32_t t;
    std::uint32_t xyp;
};
static_assert(sizeof(PackedEvent) == 8, "PackedEvent must stay 8 bytes");

inline PackedEvent pack_event(std::uint16_t x,
                              std::uint16_t y,
                              bool p,
                              std::uint32_t t) noexcept {
    return PackedEvent{
        t,
        static_cast<std::uint32_t>(x)
        | (static_cast<std::uint32_t>(y) << 11u)
        | (static_cast<std::uint32_t>(p ? 1u : 0u) << 21u)
    };
}

inline std::uint16_t event_x(const PackedEvent &e) noexcept {
    return static_cast<std::uint16_t>(e.xyp & 0x7FFu);
}

inline std::uint16_t event_y(const PackedEvent &e) noexcept {
    return static_cast<std::uint16_t>((e.xyp >> 11u) & 0x3FFu);
}

inline bool event_p(const PackedEvent &e) noexcept {
    return ((e.xyp >> 21u) & 1u) != 0;
}

} // namespace event_led_pose
