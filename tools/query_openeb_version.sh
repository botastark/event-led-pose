#!/usr/bin/env bash
set -euo pipefail

readonly image_name="localhost/event-led-pose/openeb-query:ubuntu24"
readonly containerfile="container/Containerfile.package-query"

if [[ ! -f "${containerfile}" ]]; then
    printf 'error: run this command from the repository root\n' >&2
    exit 2
fi

if ! command -v podman >/dev/null 2>&1; then
    printf 'error: podman is not installed\n' >&2
    exit 2
fi

rootless="$(podman info --format '{{.Host.Security.Rootless}}')"
if [[ "${rootless}" != "true" ]]; then
    printf 'error: podman is not running rootless\n' >&2
    exit 2
fi

printf 'Building expendable Ubuntu 24.04 package-query image...\n'
podman build \
    --pull=always \
    --tag "${image_name}" \
    --file "${containerfile}" \
    .

printf '\nAvailable OpenEB package versions:\n'
podman run \
    --rm \
    --network=none \
    --cap-drop=ALL \
    --security-opt=no-new-privileges \
    "${image_name}"

printf '\nBase-image identity:\n'
podman image inspect \
    --format 'id={{.Id}} digest={{.Digest}} created={{.Created}}' \
    docker.io/library/ubuntu:24.04

