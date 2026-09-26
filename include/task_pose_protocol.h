#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace panda_tracker {

constexpr std::uint8_t kTaskPoseVersion = 2;
constexpr std::size_t kTaskPosePacketSize = 148;

struct TaskPosePacket {
  bool valid = false;
  std::uint64_t sequence_id = 0;
  float confidence = 0.0F;
  std::array<double, 16> T_TS{};
};

enum class DecodeStatus {
  kOk = 0,
  kUnsupportedHostEndianness,
  kWrongSize,
  kWrongMagic,
  kWrongVersion,
  kInvalidValidFlag,
  kReservedFieldNonZero,
  kInvalidConfidence,
  kInvalidTransform,
};

bool host_is_little_endian();

bool is_finite_rigid_transform(
    const std::array<double, 16>& transform,
    double tolerance = 1e-6);

DecodeStatus decode_task_pose(
    const std::uint8_t* data,
    std::size_t size,
    TaskPosePacket& packet);

std::array<std::uint8_t, kTaskPosePacketSize> encode_task_pose(
    const TaskPosePacket& packet);

const char* decode_status_message(DecodeStatus status);

}  // namespace panda_tracker
