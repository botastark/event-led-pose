SHELL := /usr/bin/env bash

.PHONY: help check-camera query-openeb

help:
	@printf '%s\n' \
	  'Safe bootstrap targets:' \
	  '  make check-camera  - inspect EVK4 USB node and current permissions' \
	  '  make query-openeb  - query packaged OpenEB versions in Ubuntu 24.04'

check-camera:
	@./tools/host/check_evk4.sh

query-openeb:
	@./tools/query_openeb_version.sh

