#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd
)"

default_arguments=(
    --slice-us 20000
    --event-stride 4
    --memory-ms 250
    --detection-update-ms 40
    --display-fps 30
    --display-accumulation-us 10000
    --center-alpha 0.75
)

# Already inside the OpenEB container.
if command -v metavision_platform_info >/dev/null 2>&1; then
    exec python3 /workspace/tools/live_frequency_3led.py \
        "${default_arguments[@]}" \
        "$@"
fi

# Running on the host.
exec "$script_dir/open_camera_container.sh" \
    python3 /workspace/tools/live_frequency_3led.py \
    "${default_arguments[@]}" \
    "$@"
