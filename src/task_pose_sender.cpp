#include "task_pose_sender.h"

#include "pose_estimator.hpp"
#include "task_pose_protocol.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <array>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

namespace panda_tracker {
namespace {

std::array<double, 16> identity_transform() {
    return {
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    };
}

std::array<double, 16> pose_to_T_CT(
    const event_led_pose::PoseResult& pose)
{
    cv::Mat Rmat;
    cv::Rodrigues(pose.rvec, Rmat);

    std::array<double, 16> T_CT = {
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    };

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            T_CT[static_cast<std::size_t>(r * 4 + c)] =
                Rmat.at<double>(r, c);
        }
    }

    // Object points in PoseEstimator are expressed in millimetres,
    // therefore solveP3P returns t_CT in millimetres.
    // Robot-side tracker transforms use metres.
    T_CT[3]  = pose.position_mm[0] * 0.001;
    T_CT[7]  = pose.position_mm[1] * 0.001;
    T_CT[11] = pose.position_mm[2] * 0.001;

    return T_CT;
}

}  // namespace

TaskPoseSender::TaskPoseSender(
    const std::string& destination_ip,
    std::uint16_t destination_port)
{
    socket_fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);

    if (socket_fd_ < 0) {
        throw std::runtime_error(
            "Could not create tracker-pose UDP socket: " +
            std::string(std::strerror(errno)));
    }

    std::memset(&destination_, 0, sizeof(destination_));
    destination_.sin_family = AF_INET;
    destination_.sin_port = htons(destination_port);

    if (::inet_pton(
            AF_INET,
            destination_ip.c_str(),
            &destination_.sin_addr) != 1)
    {
        ::close(socket_fd_);
        socket_fd_ = -1;

        throw std::runtime_error(
            "Invalid tracker-pose destination IPv4 address: " +
            destination_ip);
    }
}

TaskPoseSender::~TaskPoseSender() {
    if (socket_fd_ >= 0) {
        ::close(socket_fd_);
    }
}

bool TaskPoseSender::send(
    const event_led_pose::PoseResult& pose)
{
    TaskPosePacket packet{};
    packet.sequence_id = sequence_id_++;

    if (!pose.valid) {
        // encode_task_pose() validates the matrix even for valid=false.
        packet.valid = false;
        packet.confidence = 0.0F;
        packet.T_TS = identity_transform();
    }
    else {
        packet.valid = true;
        packet.confidence = 1.0F;

        // IMPORTANT:
        // The protocol member is historically named T_TS, but the current
        // robot-side receiver interprets this payload as T_CT.
        packet.T_TS = pose_to_T_CT(pose);
    }

    const auto bytes =
        encode_task_pose(packet);

    const ssize_t sent =
        ::sendto(
            socket_fd_,
            bytes.data(),
            bytes.size(),
            MSG_DONTWAIT,
            reinterpret_cast<const sockaddr*>(&destination_),
            sizeof(destination_));

    if (sent == static_cast<ssize_t>(bytes.size())) {
        ++sent_count_;
        return true;
    }

    if (sent < 0 &&
        (errno == EAGAIN ||
         errno == EWOULDBLOCK ||
         errno == ENOBUFS))
    {
        ++dropped_count_;
        return false;
    }

    throw std::runtime_error(
        "Tracker-pose UDP send failed: " +
        std::string(std::strerror(errno)));
}

}  // namespace panda_tracker
