# !/usr/bin/env bash
export LD_LIBRARY_PATH=/workspace/.local/share/nsight_systems:$LD_LIBRARY_PATH

cd on_boarding

/workspace/.local/share/nsight_systems/nsys profile \
    --trace=cuda,osrt,nvtx \
    --output=resnet_profile \
    --force-overwrite true \
    python resnet.py "$@"