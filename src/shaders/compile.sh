#!/bin/bash
set -e

slangc shader.slang \
    -target spirv \
    -profile spirv_1_4 \
    -emit-spirv-directly \
    -fvk-use-entrypoint-name \
    -entry vertMain \
    -entry fragMain \
    -o slang.spv

echo "Compiled shader.slang -> slang.spv"
