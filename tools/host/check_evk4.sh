#!/usr/bin/env bash
set -euo pipefail

readonly expected_usb_id="04b4:00f5"

if ! command -v lsusb >/dev/null 2>&1; then
    printf 'error: lsusb is unavailable; install usbutils\n' >&2
    exit 2
fi

mapfile -t matches < <(lsusb -d "${expected_usb_id}")

if [[ "${#matches[@]}" -eq 0 ]]; then
    printf 'error: no EVK4 with USB ID %s found\n' "${expected_usb_id}" >&2
    exit 3
fi

if [[ "${#matches[@]}" -ne 1 ]]; then
    printf 'error: expected one EVK4, found %d\n' "${#matches[@]}" >&2
    printf '%s\n' "${matches[@]}" >&2
    exit 3
fi

readonly match="${matches[0]}"
bus="$(awk '{print $2}' <<<"${match}")"
device_with_colon="$(awk '{print $4}' <<<"${match}")"
device="${device_with_colon%:}"

if [[ ! "${bus}" =~ ^[0-9]{3}$ || ! "${device}" =~ ^[0-9]{3}$ ]]; then
    printf 'error: could not parse USB bus/device from: %s\n' "${match}" >&2
    exit 3
fi

readonly node="/dev/bus/usb/${bus}/${device}"

printf 'EVK4 identity:\n%s\n\n' "${match}"
printf 'Resolved device node:\n%s\n\n' "${node}"

if [[ ! -e "${node}" ]]; then
    printf 'error: resolved device node does not exist\n' >&2
    exit 3
fi

printf 'Device ownership and mode:\n'
ls -l "${node}"
stat --format='owner=%U group=%G mode=%a major_minor=%t:%T' "${node}"

printf '\nCurrent user:\n'
id

read_access=no
write_access=no
[[ -r "${node}" ]] && read_access=yes
[[ -w "${node}" ]] && write_access=yes

printf '\nCurrent access:\nread=%s write=%s\n' "${read_access}" "${write_access}"

if command -v getfacl >/dev/null 2>&1; then
    printf '\nACL:\n'
    getfacl --absolute-names "${node}"
else
    printf '\nACL: getfacl not installed; mode/ownership check completed\n'
fi

printf '\nUSB topology branch:\n'
lsusb -t

if [[ "${read_access}" == yes && "${write_access}" == yes ]]; then
    printf '\nPASS: the current user already has read/write access.\n'
    printf 'Do not install a new udev rule yet.\n'
    exit 0
fi

printf '\nNEEDS_REVIEW: the current user lacks read/write access.\n'
printf 'Do not run the camera as root and do not install a MODE=0666 rule.\n'
printf 'Send this output back so a narrow 04b4:00f5 group rule can be reviewed.\n'
exit 4

