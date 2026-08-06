#!/usr/bin/env bash
set -Eeuo pipefail

readonly USB_ID="04b4:00f5"
readonly IMAGE="${OPENEB_IMAGE:-localhost/event-led-pose/openeb-runtime:5.3.0-r3-ubuntu24}"

fail()
{
    printf 'error: %s\n' "$*" >&2
    exit 1
}

for command_name in podman lsusb xhost awk; do
    command -v "$command_name" >/dev/null 2>&1 ||
        fail "required command is missing: $command_name"
done

[[ -n "${DISPLAY:-}" ]] ||
    fail 'DISPLAY is unset. Run this from a graphical desktop terminal.'

podman image exists "$IMAGE" ||
    fail "runtime image is missing: $IMAGE"

mapfile -t cameras < <(lsusb -d "$USB_ID")

case "${#cameras[@]}" in
    0)
        fail "no EVK4 camera with USB ID $USB_ID was found"
        ;;
    1)
        ;;
    *)
        printf '%s\n' "${cameras[@]}" >&2
        fail "expected exactly one EVK4, found ${#cameras[@]}"
        ;;
esac

camera="${cameras[0]}"
bus="$(awk '{print $2}' <<<"$camera")"
device="$(awk '{gsub(":", "", $4); print $4}' <<<"$camera")"
evk_node="/dev/bus/usb/${bus}/${device}"

[[ -c "$evk_node" ]] ||
    fail "camera device node does not exist: $evk_node"

[[ -r "$evk_node" && -w "$evk_node" ]] ||
    fail "current user cannot read and write: $evk_node"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"

mkdir -p "$repo_root/data/raw" "$repo_root/data/reports"

host_user="$(id -un)"

cleanup()
{
    xhost "-SI:localuser:${host_user}" >/dev/null 2>&1 || true
}

trap cleanup EXIT

xhost "+SI:localuser:${host_user}" >/dev/null

if (( $# == 0 )); then
    container_command=(bash --noprofile --norc -i)
else
    container_command=("$@")
fi
# Preserve access to render/video devices in a rootless container.
if [[ "$(podman info --format '{{.Host.OCIRuntime.Name}}')" == "crun" ]]; then
    podman_args+=(--group-add=keep-groups)
fi

for dri_device in /dev/dri/renderD* /dev/dri/card*; do
    if [[ -e "$dri_device" ]]; then
        podman_args+=(--device="$dri_device:$dri_device")
    fi
done

printf 'Camera:    %s\n' "$camera"
printf 'USB node:  %s\n' "$evk_node"
printf 'Image:     %s\n' "$IMAGE"
printf 'Workspace: %s -> /workspace\n\n' "$repo_root"

podman run --rm -it \
    --pull=never \
    --userns=keep-id \
    --network=none \
    --cap-drop=all \
    --security-opt=no-new-privileges \
    --read-only \
    --tmpfs=/tmp:rw,nosuid,nodev,size=256m \
    --device="$evk_node" \
    --env="DISPLAY=$DISPLAY" \
    --env=HOME=/tmp \
    --env=QT_X11_NO_MITSHM=1 \
    --env='PS1=(openeb) \w \$ ' \
    --volume=/tmp/.X11-unix:/tmp/.X11-unix:ro \
    --volume="$repo_root:/workspace:rw" \
    --workdir=/workspace \
    "$IMAGE" \
    "${container_command[@]}"
