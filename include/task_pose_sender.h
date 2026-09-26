#pragma once

#include <cstdint>
#include <string>

#include <netinet/in.h>

namespace event_led_pose {
struct PoseResult;
}

namespace panda_tracker {

// Publishes the tracker PoseEstimator output directly as T_CT:
//
//   T_CT = pose of target/triangle frame T expressed in camera frame C.
//
// The existing PTP2 wire struct member is named T_TS for legacy compatibility,
// but on this robot interface the matrix payload is semantically T_CT.
//
// PoseEstimator translation is in millimetres; this sender converts it to
// metres before serialization.
class TaskPoseSender {
public:
    TaskPoseSender(
        const std::string& destination_ip,
        std::uint16_t destination_port);

    ~TaskPoseSender();

    TaskPoseSender(const TaskPoseSender&) = delete;
    TaskPoseSender& operator=(const TaskPoseSender&) = delete;

    // Sends one PTP2 packet.
    //
    // Valid pose:
    //   valid=true, confidence=1, payload=T_CT.
    //
    // Invalid pose:
    //   valid=false, confidence=0, payload=identity.
    //
    // Returns false only when a non-blocking UDP send is dropped because
    // the local socket buffer is temporarily unavailable.
    bool send(const event_led_pose::PoseResult& pose);

    std::uint64_t sent_count() const noexcept { return sent_count_; }
    std::uint64_t dropped_count() const noexcept { return dropped_count_; }

private:
    int socket_fd_ = -1;
    sockaddr_in destination_{};

    std::uint64_t sequence_id_ = 0;
    std::uint64_t sent_count_ = 0;
    std::uint64_t dropped_count_ = 0;
};

}  // namespace panda_tracker
