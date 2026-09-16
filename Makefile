SHELL := /usr/bin/env bash

BUILD_DIR := build
BUILD_TYPE ?= Release
JOBS ?= $(shell nproc)

.PHONY: help configure build rebuild run run-vis probe shell clean

help:
	@printf '%s\n' \
	  'Available targets:' \
	  '  make configure  - configure CMake build' \
	  '  make build      - build C++ targets' \
	  '  make rebuild    - clean and rebuild everything' \
	  '  make run        - run frequency filter without visualization' \
	  '  make run-vis    - run frequency filter with visualization' \
	  '  make probe      - run EVK4 stream probe' \
	  '  make shell      - open camera-enabled Podman container' \
	  '  make clean      - remove build directory'

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

build: configure
	cmake --build $(BUILD_DIR) -j$(JOBS)

rebuild:
	rm -rf $(BUILD_DIR)
	$(MAKE) build

run: build
	./$(BUILD_DIR)/live_frequency_ring

run-vis: build
	./$(BUILD_DIR)/live_frequency_ring --visualize --display-fps 120

probe: build
	./$(BUILD_DIR)/evk4_stream_probe

shell:
	./scripts/host/open_camera_container.sh

clean:
	rm -rf $(BUILD_DIR)