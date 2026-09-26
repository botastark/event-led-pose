#include "task_pose_protocol.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace panda_tracker {
namespace {

constexpr std::array<std::uint8_t, 4> kMagic{{'P', 'T', 'P', '2'}};
constexpr std::size_t kMagicOffset = 0;
constexpr std::size_t kVersionOffset = 4;
constexpr std::size_t kValidOffset = 5;
constexpr std::size_t kReservedOffset = 6;
constexpr std::size_t kSequenceOffset = 8;
constexpr std::size_t kConfidenceOffset = 16;
constexpr std::size_t kTransformOffset = 20;

template <typename T>
T load_value(const std::uint8_t* data, std::size_t offset) {
  T value{};
  std::memcpy(&value, data + offset, sizeof(T));
  return value;
}

template <typename T>
void store_value(std::uint8_t* data, std::size_t offset, const T& value) {
  std::memcpy(data + offset, &value, sizeof(T));
}

bool nearly_equal(double left, double right, double tolerance) {
  return std::abs(left - right) <= tolerance;
}

}  // namespace

bool host_is_little_endian() {
  const std::uint16_t value = 1;
  return *reinterpret_cast<const std::uint8_t*>(&value) == 1;
}

bool is_finite_rigid_transform(
    const std::array<double, 16>& transform,
    double tolerance) {
  if (!(tolerance > 0.0) || !std::isfinite(tolerance)) {
    return false;
  }

  for (const double value : transform) {
    if (!std::isfinite(value)) {
      return false;
    }
  }

  if (!nearly_equal(transform[12], 0.0, tolerance) ||
      !nearly_equal(transform[13], 0.0, tolerance) ||
      !nearly_equal(transform[14], 0.0, tolerance) ||
      !nearly_equal(transform[15], 1.0, tolerance)) {
    return false;
  }

  // Check R^T R = I for the row-major top-left 3x3 block.
  for (std::size_t column_a = 0; column_a < 3; ++column_a) {
    for (std::size_t column_b = 0; column_b < 3; ++column_b) {
      double dot = 0.0;
      for (std::size_t row = 0; row < 3; ++row) {
        dot += transform[row * 4 + column_a] *
               transform[row * 4 + column_b];
      }
      const double expected = column_a == column_b ? 1.0 : 0.0;
      if (!nearly_equal(dot, expected, tolerance)) {
        return false;
      }
    }
  }

  const double determinant =
      transform[0] * (transform[5] * transform[10] -
                      transform[6] * transform[9]) -
      transform[1] * (transform[4] * transform[10] -
                      transform[6] * transform[8]) +
      transform[2] * (transform[4] * transform[9] -
                      transform[5] * transform[8]);

  return nearly_equal(determinant, 1.0, tolerance);
}

DecodeStatus decode_task_pose(
    const std::uint8_t* data,
    std::size_t size,
    TaskPosePacket& packet) {
  if (!host_is_little_endian()) {
    return DecodeStatus::kUnsupportedHostEndianness;
  }
  if (data == nullptr || size != kTaskPosePacketSize) {
    return DecodeStatus::kWrongSize;
  }
  if (!std::equal(kMagic.begin(), kMagic.end(), data + kMagicOffset)) {
    return DecodeStatus::kWrongMagic;
  }
  if (data[kVersionOffset] != kTaskPoseVersion) {
    return DecodeStatus::kWrongVersion;
  }

  const std::uint8_t valid_value = data[kValidOffset];
  if (valid_value > 1) {
    return DecodeStatus::kInvalidValidFlag;
  }
  if (load_value<std::uint16_t>(data, kReservedOffset) != 0) {
    return DecodeStatus::kReservedFieldNonZero;
  }

  TaskPosePacket decoded{};
  decoded.sequence_id = load_value<std::uint64_t>(data, kSequenceOffset);
  decoded.confidence = load_value<float>(data, kConfidenceOffset);
  decoded.valid = valid_value == 1;

  if (!std::isfinite(decoded.confidence) ||
      decoded.confidence < 0.0F || decoded.confidence > 1.0F) {
    return DecodeStatus::kInvalidConfidence;
  }

  for (std::size_t index = 0; index < decoded.T_TS.size(); ++index) {
    decoded.T_TS[index] = load_value<double>(
        data,
        kTransformOffset + index * sizeof(double));
  }

  if (!is_finite_rigid_transform(decoded.T_TS)) {
    return DecodeStatus::kInvalidTransform;
  }

  packet = decoded;
  return DecodeStatus::kOk;
}

std::array<std::uint8_t, kTaskPosePacketSize> encode_task_pose(
    const TaskPosePacket& packet) {
  if (!host_is_little_endian()) {
    throw std::runtime_error(
        "PTP2 encoding currently requires a little-endian host.");
  }
  if (!std::isfinite(packet.confidence) ||
      packet.confidence < 0.0F || packet.confidence > 1.0F) {
    throw std::invalid_argument("Task-pose confidence must be in [0, 1].");
  }
  if (!is_finite_rigid_transform(packet.T_TS)) {
    throw std::invalid_argument("T_TS must be a finite rigid transform.");
  }

  std::array<std::uint8_t, kTaskPosePacketSize> data{};
  std::copy(kMagic.begin(), kMagic.end(), data.begin() + kMagicOffset);
  data[kVersionOffset] = kTaskPoseVersion;
  data[kValidOffset] = packet.valid ? 1 : 0;
  store_value<std::uint16_t>(data.data(), kReservedOffset, 0);
  store_value<std::uint64_t>(
      data.data(), kSequenceOffset, packet.sequence_id);
  store_value<float>(data.data(), kConfidenceOffset, packet.confidence);

  for (std::size_t index = 0; index < packet.T_TS.size(); ++index) {
    store_value<double>(
        data.data(),
        kTransformOffset + index * sizeof(double),
        packet.T_TS[index]);
  }

  return data;
}

const char* decode_status_message(DecodeStatus status) {
  switch (status) {
    case DecodeStatus::kOk:
      return "ok";
    case DecodeStatus::kUnsupportedHostEndianness:
      return "unsupported_host_endianness";
    case DecodeStatus::kWrongSize:
      return "wrong_size";
    case DecodeStatus::kWrongMagic:
      return "wrong_magic";
    case DecodeStatus::kWrongVersion:
      return "wrong_version";
    case DecodeStatus::kInvalidValidFlag:
      return "invalid_valid_flag";
    case DecodeStatus::kReservedFieldNonZero:
      return "reserved_field_nonzero";
    case DecodeStatus::kInvalidConfidence:
      return "invalid_confidence";
    case DecodeStatus::kInvalidTransform:
      return "invalid_transform";
  }
  return "unknown";
}

}  // namespace panda_tracker

