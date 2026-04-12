#!/usr/bin/env bash
set -e

cmake --workflow --preset "${1:-debug}"
./_build/main
